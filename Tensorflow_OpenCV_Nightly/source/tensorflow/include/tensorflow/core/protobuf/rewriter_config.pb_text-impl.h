// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_protobuf_rewriter_config_proto_IMPL_H_
#define tensorflow_core_protobuf_rewriter_config_proto_IMPL_H_

#include "tensorflow/core/lib/strings/proto_text_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb_text.h"

namespace tensorflow {

namespace internal {

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::AutoParallelOptions& msg);
bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::AutoParallelOptions* msg);

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::RewriterConfig& msg);
bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::RewriterConfig* msg);

}  // namespace internal

}  // namespace tensorflow

#endif  // tensorflow_core_protobuf_rewriter_config_proto_IMPL_H_
