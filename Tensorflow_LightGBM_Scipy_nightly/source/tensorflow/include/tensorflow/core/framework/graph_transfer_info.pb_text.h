// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_framework_graph_transfer_info_proto_H_
#define tensorflow_core_framework_graph_transfer_info_proto_H_

#include "tensorflow/core/framework/graph_transfer_info.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Enum text output for tensorflow.GraphTransferInfo.Destination
const char* EnumName_GraphTransferInfo_Destination(
    ::tensorflow::GraphTransferInfo_Destination value);

// Message-text conversion for tensorflow.GraphTransferInfo.NodeInput
string ProtoDebugString(
    const ::tensorflow::GraphTransferInfo_NodeInput& msg);
string ProtoShortDebugString(
    const ::tensorflow::GraphTransferInfo_NodeInput& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::GraphTransferInfo_NodeInput* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.GraphTransferInfo.NodeInfo
string ProtoDebugString(
    const ::tensorflow::GraphTransferInfo_NodeInfo& msg);
string ProtoShortDebugString(
    const ::tensorflow::GraphTransferInfo_NodeInfo& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::GraphTransferInfo_NodeInfo* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.GraphTransferInfo.ConstNodeInfo
string ProtoDebugString(
    const ::tensorflow::GraphTransferInfo_ConstNodeInfo& msg);
string ProtoShortDebugString(
    const ::tensorflow::GraphTransferInfo_ConstNodeInfo& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::GraphTransferInfo_ConstNodeInfo* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.GraphTransferInfo.NodeInputInfo
string ProtoDebugString(
    const ::tensorflow::GraphTransferInfo_NodeInputInfo& msg);
string ProtoShortDebugString(
    const ::tensorflow::GraphTransferInfo_NodeInputInfo& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::GraphTransferInfo_NodeInputInfo* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.GraphTransferInfo.NodeOutputInfo
string ProtoDebugString(
    const ::tensorflow::GraphTransferInfo_NodeOutputInfo& msg);
string ProtoShortDebugString(
    const ::tensorflow::GraphTransferInfo_NodeOutputInfo& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::GraphTransferInfo_NodeOutputInfo* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.GraphTransferInfo.GraphInputNodeInfo
string ProtoDebugString(
    const ::tensorflow::GraphTransferInfo_GraphInputNodeInfo& msg);
string ProtoShortDebugString(
    const ::tensorflow::GraphTransferInfo_GraphInputNodeInfo& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::GraphTransferInfo_GraphInputNodeInfo* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.GraphTransferInfo.GraphOutputNodeInfo
string ProtoDebugString(
    const ::tensorflow::GraphTransferInfo_GraphOutputNodeInfo& msg);
string ProtoShortDebugString(
    const ::tensorflow::GraphTransferInfo_GraphOutputNodeInfo& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::GraphTransferInfo_GraphOutputNodeInfo* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.GraphTransferInfo
string ProtoDebugString(
    const ::tensorflow::GraphTransferInfo& msg);
string ProtoShortDebugString(
    const ::tensorflow::GraphTransferInfo& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::GraphTransferInfo* msg)
        TF_MUST_USE_RESULT;

}  // namespace tensorflow

#endif  // tensorflow_core_framework_graph_transfer_info_proto_H_
