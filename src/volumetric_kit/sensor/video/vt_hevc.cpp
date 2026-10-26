// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "vt_hevc.hpp"

#include <algorithm>

namespace volumetric_kit::recon::sensor::video {
namespace {

// Stores @p nal's bytes under @p id: whether they differ from what was there.
bool store(std::map<std::uint32_t, std::vector<std::uint8_t>>& sets,
           std::uint32_t id, const hevc::Nal& nal) {
  std::vector<std::uint8_t>& stored = sets[id];
  if (stored.size() == nal.size &&
      std::equal(stored.begin(), stored.end(), nal.data)) {
    return false;
  }
  stored.assign(nal.data, nal.data + nal.size);
  return true;
}

std::string format_name(const hevc::Sps& sps) {
  static const char* const kChroma[] = {"4:0:0", "4:2:0", "4:2:2", "4:4:4"};
  return std::to_string(std::max(sps.bit_depth_luma, sps.bit_depth_chroma)) +
         "-bit " + kChroma[sps.chroma_format_idc];
}

}  // namespace

core::Result<std::unique_ptr<VtHevc>> VtHevc::open(
    const core::Device& device, const char* who,
    std::optional<VideoColorDescription> unlabelled_color,
    const std::uint32_t* queue_families, std::uint32_t queue_family_count) {
  if (!VTIsHardwareDecodeSupported(kCMVideoCodecType_HEVC)) {
    return core::Status::unsupported(
        std::string(who) + ": VideoToolbox has no hardware HEVC decoder");
  }
  VKC_ASSIGN(PixelBufferImporter pictures,
             PixelBufferImporter::create(device, who, queue_families,
                                         queue_family_count));
  return std::unique_ptr<VtHevc>(
      new VtHevc(std::move(pictures), who, std::move(unlabelled_color)));
}

core::Status VtHevc::io_error(const std::string& why) const {
  return core::Status::io_error(std::string(who_) + ": " + why);
}

core::Status VtHevc::unsupported(const std::string& why) const {
  return core::Status::unsupported(std::string(who_) + ": " + why);
}

core::Status VtHevc::refuse(const core::Status& refusal) {
  show_all();
  refusal_ = refusal.message();
  return refusal;
}

void VtHevc::show_next() {
  const auto first = std::min_element(
      held_.begin(), held_.end(),
      [](const Held& a, const Held& b) { return a.poc < b.poc; });
  ready_.push_back(std::move(*first));
  held_.erase(first);
}

void VtHevc::show_all() {
  while (!held_.empty()) show_next();
}

core::Status VtHevc::store_parameter_set(const hevc::Nal& nal) {
  // The reader's refusals are IoError: bytes that do not parse.
  if (nal.type == hevc::kVps) {
    const core::Result<std::uint32_t> id = hevc::vps_id(nal);
    if (!id) return id.status().with_context(who_);
    if (store(vps_bytes_, *id, nal)) ++params_version_;
  } else if (nal.type == hevc::kSps) {
    // Read whole before it can refuse the stream: bytes that do not parse
    // cost this access unit, not the stream.
    const core::Result<hevc::Sps> sps = hevc::parse_sps(nal);
    if (!sps) return sps.status().with_context(who_);
    if (sps->chroma_format_idc != 1 || sps->bit_depth_luma != 8 ||
        sps->bit_depth_chroma != 8) {
      return refuse(unsupported("the device path takes 8-bit 4:2:0 only, not " +
                                format_name(*sps)));
    }
    // VideoToolbox decodes the coded picture's corner, not the window.
    if (sps->crop_left != 0 || sps->crop_top != 0) {
      return refuse(
          unsupported("VideoToolbox cannot crop the left or top of a picture"));
    }
    if (store(sps_bytes_, sps->id, nal)) ++params_version_;
    sps_[sps->id] = *sps;
  } else {
    const core::Result<hevc::Pps> pps = hevc::parse_pps(nal);
    if (!pps) return pps.status().with_context(who_);
    if (store(pps_bytes_, pps->id, nal)) ++params_version_;
    pps_[pps->id] = *pps;
  }
  return {};
}

core::Status VtHevc::ready_session(const hevc::Sps& sps) {
  if (session_ != nullptr && format_version_ == params_version_ &&
      format_sps_ == sps.id) {
    return {};
  }
  // The sequence's VPS and SPS, and every PPS on that SPS.
  const auto vps = vps_bytes_.find(sps.vps_id);
  if (vps == vps_bytes_.end()) return io_error("an SPS whose VPS was not sent");
  std::vector<const std::uint8_t*> sets = {vps->second.data(),
                                           sps_bytes_.at(sps.id).data()};
  std::vector<std::size_t> sizes = {vps->second.size(),
                                    sps_bytes_.at(sps.id).size()};
  for (const auto& [id, pps] : pps_) {
    if (pps.sps_id != sps.id) continue;
    sets.push_back(pps_bytes_.at(id).data());
    sizes.push_back(pps_bytes_.at(id).size());
  }
  CMVideoFormatDescriptionRef described = nullptr;
  const OSStatus status = CMVideoFormatDescriptionCreateFromHEVCParameterSets(
      kCFAllocatorDefault, sets.size(), sets.data(), sizes.data(), 4, nullptr,
      &described);
  if (status == kCMFormatDescriptionError_AllocationFailed) {
    return vt_error(who_, "describing the stream", status);
  }
  if (status != noErr) {
    return io_error("parameter sets VideoToolbox does not take (status " +
                    std::to_string(status) + ")");
  }
  FormatPtr format(described);
  const bool full_range = sps.color.full_range;
  // A session takes a new format of the same sequence where it can; a new
  // range, or one it cannot take, needs a session of its own.
  if (session_ == nullptr || full_range != format_full_range_ ||
      !VTDecompressionSessionCanAcceptFormatDescription(session_.get(),
                                                        format.get())) {
    session_.reset();
    VKC_ASSIGN(session_,
               open_session(format.get(), full_range, who_, "an HEVC session"));
    format_full_range_ = full_range;
  }
  format_ = std::move(format);
  format_version_ = params_version_;
  format_sps_ = sps.id;
  return {};
}

core::Status VtHevc::decode(const std::vector<hevc::Nal>& slices,
                            std::int64_t pts) {
  const core::Result<hevc::SliceStart> slice =
      hevc::parse_slice_start(slices.front(), pps_, sps_);
  if (!slice) return slice.status().with_context(who_);
  if (!slice->first_slice_segment) {
    return io_error("an access unit that does not start its picture");
  }
  for (std::size_t i = 1; i < slices.size(); ++i) {
    // first_slice_segment_in_pic_flag
    if (slices[i].size > 2 && (slices[i].data[2] & 0x80) != 0) {
      return io_error("an access unit of more than one picture");
    }
  }
  const hevc::Sps& sps = sps_.at(pps_.at(slice->pps_id).sps_id);
  const int type = slices.front().type;
  const bool irap = hevc::is_irap(type);

  // 8.1.3: an IDR, a BLA or the first key frame decoded starts the sequence
  // afresh; the leading pictures of that one are predicted from pictures the
  // decoder never had, and are skipped.
  if (!irap && first_picture_) {
    return io_error("a picture predicted from before the first key frame");
  }
  const bool restart =
      irap && (first_picture_ || hevc::is_idr(type) || type <= 18);
  if (irap) skip_rasl_ = restart;
  if (hevc::is_rasl(type) && skip_rasl_) return {};

  // 8.3.1: the picture order count, its most significant part carried on
  // from the last picture of sub-layer 0 that others can refer to.
  const std::int32_t max_lsb = std::int32_t{1} << sps.log2_max_poc_lsb;
  const auto lsb = static_cast<std::int32_t>(slice->poc_lsb);
  std::int32_t msb = 0;
  if (!restart) {
    const std::int32_t prev_lsb = prev_tid0_poc_ & (max_lsb - 1);
    const std::int32_t prev_msb = prev_tid0_poc_ - prev_lsb;
    if (lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2) {
      msb = prev_msb + max_lsb;
    } else if (lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2) {
      msb = prev_msb - max_lsb;
    } else {
      msb = prev_msb;
    }
  }
  const std::int32_t poc = msb + lsb;
  const bool radl = type == 6 || type == 7;
  const bool sub_layer_non_reference = type <= 14 && type % 2 == 0;
  if (slices.front().temporal_id == 0 && !hevc::is_rasl(type) && !radl &&
      !sub_layer_non_reference) {
    prev_tid0_poc_ = poc;
  }

  // C.5.2.2: a sequence started afresh shows the pictures before it, unless
  // it says to drop them.
  if (restart) {
    if (slice->no_output_of_prior_pics) {
      held_.clear();
    } else {
      show_all();
    }
  }
  first_picture_ = false;

  if (core::Status ready = ready_session(sps); !ready.ok()) {
    if (ready.domain() == core::Status::Code::Unsupported) return refuse(ready);
    return ready;
  }
  // The picture's slices, each behind its length, as the format says.
  sample_.clear();
  for (const hevc::Nal& nal : slices) {
    const auto n = static_cast<std::uint32_t>(nal.size);
    const std::uint8_t length[4] = {
        static_cast<std::uint8_t>(n >> 24), static_cast<std::uint8_t>(n >> 16),
        static_cast<std::uint8_t>(n >> 8), static_cast<std::uint8_t>(n)};
    sample_.insert(sample_.end(), length, length + 4);
    sample_.insert(sample_.end(), nal.data, nal.data + nal.size);
  }
  core::Result<PixelBufferPtr> decoded =
      decode_sample(session_.get(), format_.get(), sample_.data(),
                    sample_.size(), who_, "a picture");
  if (!decoded) {
    const core::Status::Code code = decoded.status().domain();
    if (code == core::Status::Code::Unsupported) {
      return refuse(decoded.status());
    }
    // Bad data, a missing reference among it, costs the pictures up to the
    // next key frame, which VideoToolbox reports as they come. Any other
    // failure takes a session afresh, from the next key frame.
    if (code != core::Status::Code::IoError) {
      session_.reset();
      first_picture_ = true;
    }
    return decoded.status();
  }
  if (decoded.value() == nullptr)
    return io_error("the picture does not decode");

  // C.5.2.3: shown once more pictures wait than the sequence reorders.
  if (!slice->pic_output) return {};
  Held held;
  held.poc = poc;
  held.pts = pts;
  held.pixels = std::move(decoded).value();
  held.width = sps.width();
  held.height = sps.height();
  held.color = sps.color;
  held_.push_back(std::move(held));
  while (held_.size() > sps.max_num_reorder) show_next();
  return {};
}

core::Status VtHevc::send(const std::uint8_t* data, std::size_t size,
                          std::int64_t pts) {
  if (!refusal_.empty()) return core::Status::unsupported(refusal_);
  if (ended_) {
    return core::Status::invalid_argument(
        std::string(who_) + ": data sent after the end of the stream");
  }
  if (!ready_.empty()) {
    return core::Status::invalid_argument(
        std::string(who_) + ": take the pictures waiting before sending more");
  }
  if (size == 0) {
    ended_ = true;
    show_all();
    return {};
  }
  if (data == nullptr) {
    return core::Status::invalid_argument(std::string(who_) +
                                          ": an access unit needs data");
  }
  if (core::Status split = hevc::split_annex_b(data, size, nals_); !split) {
    return std::move(split).with_context(who_);
  }
  // In order: the parameter sets first, the picture's slices once all are
  // in, and an end of sequence after them.
  slices_.clear();
  bool end_of_sequence = false;
  for (const hevc::Nal& nal : nals_) {
    if (nal.layer_id != 0) continue;  // the base layer alone
    if (nal.type >= hevc::kVps && nal.type <= hevc::kPps) {
      VKC_TRY(store_parameter_set(nal));
    } else if (nal.type == hevc::kEos) {
      end_of_sequence = true;
    } else if (nal.type <= 9 || (nal.type >= 16 && nal.type <= 21)) {
      slices_.push_back(nal);  // a VCL type the standard defines
    }
  }
  if (!slices_.empty()) VKC_TRY(decode(slices_, pts));
  if (end_of_sequence) {
    show_all();
    first_picture_ = true;
  }
  return {};
}

core::Result<std::optional<DecodedPicture>> VtHevc::receive() {
  if (ready_.empty()) {
    // Pictures decoded before a refusal still come out; then the refusal.
    if (!refusal_.empty()) return core::Status::unsupported(refusal_);
    return std::optional<DecodedPicture>();
  }
  Held held = std::move(ready_.front());
  ready_.pop_front();
  // The images hold the picture's buffer, ours let go on the way out.
  DecodedPicture out;
  if (core::Status imported =
          pictures_.import(held.pixels.get(), held.width, held.height, out.yuv);
      !imported.ok()) {
    // A picture the device path cannot take: the next ones would be the
    // same.
    if (imported.domain() == core::Status::Code::Unsupported) {
      held_.clear();
      ready_.clear();
      refusal_ = imported.message();
    }
    return imported;
  }
  describe_color(held.color, unlabelled_color_, out);
  out.pts = held.pts;
  return std::optional<DecodedPicture>(std::move(out));
}

void VtHevc::reset() noexcept {
  // As for a seek: what is held is dropped, the end undone, and the next key
  // frame decoded as the first, on a session of its own.
  held_.clear();
  ready_.clear();
  ended_ = false;
  first_picture_ = true;
  session_.reset();
}

}  // namespace volumetric_kit::recon::sensor::video
