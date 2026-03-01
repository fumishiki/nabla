//! ONNX export for nabla modules.
//!
//! Implements P6-ONNX-01 through P6-ONNX-05: Module trait walk to ONNX DAG,
//! minimal protobuf serialization, opset 21 compliance, dynamic axes support,
//! and onnxruntime verification (documented, not a runtime dependency).

use std::io::{self, Write};
use std::path::Path;

use nabla_core::backend::Backend;
use nabla_core::scalar::Scalar;

use crate::ml::module::Module;

// --- Protobuf wire types and encoding (minimal subset for ONNX) ---

const WIRE_VARINT: u8 = 0;
const WIRE_LEN: u8 = 2;
const WIRE_FIXED32: u8 = 5;
const _WIRE_FIXED64: u8 = 1;

fn encode_varint(buf: &mut Vec<u8>, mut val: u64) {
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if val == 0 { buf.push(byte); return; }
        buf.push(byte | 0x80);
    }
}

fn encode_tag(buf: &mut Vec<u8>, field: u32, wire: u8) {
    encode_varint(buf, u64::from(field) << 3 | u64::from(wire));
}

fn encode_varint_field(buf: &mut Vec<u8>, field: u32, val: u64) {
    if val == 0 { return; }
    encode_tag(buf, field, WIRE_VARINT);
    encode_varint(buf, val);
}

fn encode_bytes_field(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
    encode_tag(buf, field, WIRE_LEN);
    encode_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

fn encode_string_field(buf: &mut Vec<u8>, field: u32, s: &str) {
    if s.is_empty() { return; }
    encode_bytes_field(buf, field, s.as_bytes());
}

fn encode_submessage_field(buf: &mut Vec<u8>, field: u32, sub: &[u8]) {
    encode_bytes_field(buf, field, sub);
}

fn encode_float_field(buf: &mut Vec<u8>, field: u32, val: f32) {
    encode_tag(buf, field, WIRE_FIXED32);
    buf.extend_from_slice(&val.to_le_bytes());
}

fn encode_int64_field(buf: &mut Vec<u8>, field: u32, val: i64) {
    if val == 0 { return; }
    encode_tag(buf, field, WIRE_VARINT);
    encode_varint(buf, val as u64);
}

fn encode_packed_i64(buf: &mut Vec<u8>, field: u32, vals: &[i64]) {
    if vals.is_empty() { return; }
    let mut inner = Vec::new();
    for &v in vals { encode_varint(&mut inner, v as u64); }
    encode_bytes_field(buf, field, &inner);
}

fn encode_packed_f32(buf: &mut Vec<u8>, field: u32, vals: &[f32]) {
    if vals.is_empty() { return; }
    let mut inner = Vec::with_capacity(vals.len() * 4);
    for &v in vals { inner.extend_from_slice(&v.to_le_bytes()); }
    encode_bytes_field(buf, field, &inner);
}

// --- ONNX data types ---

const ONNX_FLOAT: i32 = 1;

// --- ONNX protobuf field numbers ---
// ModelProto: ir_version=1, opset_import=8, graph=7
// OpsetImport: domain=1, version=2
// GraphProto: node=1, name=2, input=11, output=12, initializer=5
// NodeProto: input=1, output=2, name=3, op_type=4, domain=7, attribute=5
// AttributeProto: name=1, type=2, f=4, i=3, s=3(?), t=5, ints=8, floats=7, strings=9
// TensorProto: dims=1, data_type=2, float_data=4, name=8, raw_data=13
// ValueInfoProto: name=1, type=2
// TypeProto: tensor_type=1
// TypeProto.Tensor: elem_type=1, shape=2
// TensorShapeProto: dim=1
// TensorShapeProto.Dimension: dim_value=1, dim_param=2

// --- ONNX operator mapping ---

/// ONNX operator kind for nabla→ONNX conversion.
#[derive(Debug, Clone)]
pub enum OnnxOp {
    /// Matrix multiply: C = A @ B
    MatMul,
    /// Gemm: Y = alpha*A@B + beta*C (for linear layers with bias)
    Gemm { alpha: f32, beta: f32, trans_b: bool },
    /// Element-wise add
    Add,
    /// Relu activation
    Relu,
    /// Gelu activation (opset 20+)
    Gelu,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Softmax along an axis
    Softmax { axis: i64 },
    /// Layer normalization (opset 17+)
    LayerNormalization { axis: i64, epsilon: f32 },
    /// Dropout (identity in eval mode)
    Dropout,
    /// Reshape
    Reshape,
    /// Transpose with perm
    Transpose { perm: Vec<i64> },
    /// Conv (1d/2d/3d)
    Conv { kernel_shape: Vec<i64>, strides: Vec<i64>, pads: Vec<i64>, group: i64 },
    /// Gather (embedding lookup)
    Gather { axis: i64 },
    /// Custom op for extensibility
    Custom { op_type: String, attributes: Vec<OnnxAttr> },
}

/// ONNX attribute value.
#[derive(Debug, Clone)]
pub enum OnnxAttr {
    /// Float attribute
    Float(String, f32),
    /// Integer attribute
    Int(String, i64),
    /// List of integers
    Ints(String, Vec<i64>),
    /// List of floats
    Floats(String, Vec<f32>),
}

/// A node in the ONNX computational graph.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Human-readable node name
    pub name: String,
    /// ONNX operator
    pub op: OnnxOp,
}

/// Dimension specification for dynamic axes.
#[derive(Debug, Clone)]
pub enum DimSpec {
    /// Fixed integer dimension
    Fixed(i64),
    /// Symbolic/dynamic dimension (e.g. "batch_size", "seq_len")
    Dynamic(String),
}

/// Description of a graph input/output tensor.
#[derive(Debug, Clone)]
pub struct TensorSpec {
    /// Tensor name
    pub name: String,
    /// Dimensions (fixed or dynamic)
    pub dims: Vec<DimSpec>,
    /// Element data type (ONNX enum)
    pub elem_type: i32,
}

impl TensorSpec {
    /// Create a spec with f32 element type.
    #[must_use]
    pub fn float(name: impl Into<String>, dims: Vec<DimSpec>) -> Self {
        Self { name: name.into(), dims, elem_type: ONNX_FLOAT }
    }
}

/// ONNX weight initializer (named tensor data).
#[derive(Debug, Clone)]
pub struct OnnxInitializer {
    /// Parameter name
    pub name: String,
    /// Shape dimensions
    pub dims: Vec<i64>,
    /// Flattened f32 data
    pub data: Vec<f32>,
}

/// Complete ONNX graph ready for serialization.
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    /// Graph name
    pub name: String,
    /// Computation nodes in topological order
    pub nodes: Vec<OnnxNode>,
    /// Graph-level inputs (excluding initializers)
    pub inputs: Vec<TensorSpec>,
    /// Graph-level outputs
    pub outputs: Vec<TensorSpec>,
    /// Weight initializers
    pub initializers: Vec<OnnxInitializer>,
}

/// ONNX model container.
#[derive(Debug, Clone)]
pub struct OnnxModel {
    /// IR version (default 9 for opset 21)
    pub ir_version: i64,
    /// Opset version (default 21)
    pub opset_version: i64,
    /// Producer name
    pub producer: String,
    /// Model graph
    pub graph: OnnxGraph,
}

impl Default for OnnxModel {
    fn default() -> Self {
        Self {
            ir_version: 9,
            opset_version: 21,
            producer: "nabla".to_owned(),
            graph: OnnxGraph {
                name: "nabla_model".to_owned(),
                nodes: Vec::new(), inputs: Vec::new(),
                outputs: Vec::new(), initializers: Vec::new(),
            },
        }
    }
}

// --- Protobuf serialization for ONNX types ---

fn encode_onnx_attr(attr: &OnnxAttr) -> Vec<u8> {
    let mut buf = Vec::new();
    match attr {
        OnnxAttr::Float(name, v) => {
            encode_string_field(&mut buf, 1, name); // name
            encode_varint_field(&mut buf, 2, 1); // type=FLOAT
            encode_float_field(&mut buf, 4, *v); // f
        }
        OnnxAttr::Int(name, v) => {
            encode_string_field(&mut buf, 1, name); // name
            encode_varint_field(&mut buf, 2, 2); // type=INT
            encode_int64_field(&mut buf, 3, *v); // i
        }
        OnnxAttr::Ints(name, vals) => {
            encode_string_field(&mut buf, 1, name); // name
            encode_varint_field(&mut buf, 2, 7); // type=INTS
            encode_packed_i64(&mut buf, 8, vals); // ints
        }
        OnnxAttr::Floats(name, vals) => {
            encode_string_field(&mut buf, 1, name); // name
            encode_varint_field(&mut buf, 2, 6); // type=FLOATS
            encode_packed_f32(&mut buf, 7, vals); // floats
        }
    }
    buf
}

fn encode_op_attributes(op: &OnnxOp) -> Vec<Vec<u8>> {
    match op {
        OnnxOp::Gemm { alpha, beta, trans_b } => {
            let mut attrs = vec![
                encode_onnx_attr(&OnnxAttr::Float("alpha".to_owned(), *alpha)),
                encode_onnx_attr(&OnnxAttr::Float("beta".to_owned(), *beta)),
            ];
            if *trans_b {
                attrs.push(encode_onnx_attr(&OnnxAttr::Int("transB".to_owned(), 1)));
            }
            attrs
        }
        OnnxOp::Softmax { axis } => {
            vec![encode_onnx_attr(&OnnxAttr::Int("axis".to_owned(), *axis))]
        }
        OnnxOp::LayerNormalization { axis, epsilon } => {
            vec![
                encode_onnx_attr(&OnnxAttr::Int("axis".to_owned(), *axis)),
                encode_onnx_attr(&OnnxAttr::Float("epsilon".to_owned(), *epsilon)),
            ]
        }
        OnnxOp::Transpose { perm } => {
            vec![encode_onnx_attr(&OnnxAttr::Ints("perm".to_owned(), perm.clone()))]
        }
        OnnxOp::Conv { kernel_shape, strides, pads, group } => {
            let mut attrs = vec![
                encode_onnx_attr(&OnnxAttr::Ints("kernel_shape".to_owned(), kernel_shape.clone())),
                encode_onnx_attr(&OnnxAttr::Ints("strides".to_owned(), strides.clone())),
                encode_onnx_attr(&OnnxAttr::Ints("pads".to_owned(), pads.clone())),
            ];
            if *group > 1 {
                attrs.push(encode_onnx_attr(&OnnxAttr::Int("group".to_owned(), *group)));
            }
            attrs
        }
        OnnxOp::Gather { axis } => {
            vec![encode_onnx_attr(&OnnxAttr::Int("axis".to_owned(), *axis))]
        }
        OnnxOp::Custom { attributes, .. } => {
            attributes.iter().map(encode_onnx_attr).collect()
        }
        OnnxOp::MatMul | OnnxOp::Add | OnnxOp::Relu | OnnxOp::Gelu
        | OnnxOp::Sigmoid | OnnxOp::Tanh | OnnxOp::Dropout
        | OnnxOp::Reshape => Vec::new(),
    }
}

fn op_type_str(op: &OnnxOp) -> &str {
    match op {
        OnnxOp::MatMul => "MatMul", OnnxOp::Gemm { .. } => "Gemm",
        OnnxOp::Add => "Add", OnnxOp::Relu => "Relu",
        OnnxOp::Gelu => "Gelu", OnnxOp::Sigmoid => "Sigmoid",
        OnnxOp::Tanh => "Tanh", OnnxOp::Softmax { .. } => "Softmax",
        OnnxOp::LayerNormalization { .. } => "LayerNormalization",
        OnnxOp::Dropout => "Dropout", OnnxOp::Reshape => "Reshape",
        OnnxOp::Transpose { .. } => "Transpose",
        OnnxOp::Conv { .. } => "Conv",
        OnnxOp::Gather { .. } => "Gather",
        OnnxOp::Custom { op_type, .. } => op_type.as_str(),
    }
}

fn encode_node(node: &OnnxNode) -> Vec<u8> {
    let mut buf = Vec::new();
    for inp in &node.inputs { encode_string_field(&mut buf, 1, inp); }
    for out in &node.outputs { encode_string_field(&mut buf, 2, out); }
    encode_string_field(&mut buf, 3, &node.name);
    encode_string_field(&mut buf, 4, op_type_str(&node.op));
    for attr_bytes in encode_op_attributes(&node.op) {
        encode_submessage_field(&mut buf, 5, &attr_bytes);
    }
    buf
}

fn encode_dim(dim: &DimSpec) -> Vec<u8> {
    let mut buf = Vec::new();
    match dim {
        DimSpec::Fixed(v) => encode_int64_field(&mut buf, 1, *v),
        DimSpec::Dynamic(s) => encode_string_field(&mut buf, 2, s),
    }
    buf
}

fn encode_tensor_shape(dims: &[DimSpec]) -> Vec<u8> {
    let mut buf = Vec::new();
    for d in dims {
        let dim_bytes = encode_dim(d);
        encode_submessage_field(&mut buf, 1, &dim_bytes);
    }
    buf
}

fn encode_tensor_type(elem_type: i32, dims: &[DimSpec]) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_varint_field(&mut buf, 1, elem_type as u64); // elem_type
    let shape_bytes = encode_tensor_shape(dims);
    encode_submessage_field(&mut buf, 2, &shape_bytes); // shape
    buf
}

fn encode_type_proto(elem_type: i32, dims: &[DimSpec]) -> Vec<u8> {
    let mut buf = Vec::new();
    let tensor_type = encode_tensor_type(elem_type, dims);
    encode_submessage_field(&mut buf, 1, &tensor_type); // tensor_type (field 1)
    buf
}

fn encode_value_info(spec: &TensorSpec) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_string_field(&mut buf, 1, &spec.name);
    let type_proto = encode_type_proto(spec.elem_type, &spec.dims);
    encode_submessage_field(&mut buf, 2, &type_proto);
    buf
}

fn encode_tensor_proto(init: &OnnxInitializer) -> Vec<u8> {
    let mut buf = Vec::new();
    // dims (field 1, packed int64)
    encode_packed_i64(&mut buf, 1, &init.dims);
    // data_type (field 2)
    encode_varint_field(&mut buf, 2, ONNX_FLOAT as u64);
    // raw_data (field 13) — more compact than float_data
    let mut raw = Vec::with_capacity(init.data.len() * 4);
    for &v in &init.data { raw.extend_from_slice(&v.to_le_bytes()); }
    encode_bytes_field(&mut buf, 13, &raw);
    // name (field 8)
    encode_string_field(&mut buf, 8, &init.name);
    buf
}

fn encode_graph(graph: &OnnxGraph) -> Vec<u8> {
    let mut buf = Vec::new();
    // nodes (field 1)
    for node in &graph.nodes {
        let node_bytes = encode_node(node);
        encode_submessage_field(&mut buf, 1, &node_bytes);
    }
    // name (field 2)
    encode_string_field(&mut buf, 2, &graph.name);
    // initializer (field 5)
    for init in &graph.initializers {
        let init_bytes = encode_tensor_proto(init);
        encode_submessage_field(&mut buf, 5, &init_bytes);
    }
    // input (field 11) — includes initializers as inputs per ONNX spec
    for inp in &graph.inputs {
        let vi = encode_value_info(inp);
        encode_submessage_field(&mut buf, 11, &vi);
    }
    for init in &graph.initializers {
        let dims: Vec<DimSpec> = init.dims.iter().map(|&d| DimSpec::Fixed(d)).collect();
        let spec = TensorSpec { name: init.name.clone(), dims, elem_type: ONNX_FLOAT };
        let vi = encode_value_info(&spec);
        encode_submessage_field(&mut buf, 11, &vi);
    }
    // output (field 12)
    for out in &graph.outputs {
        let vi = encode_value_info(out);
        encode_submessage_field(&mut buf, 12, &vi);
    }
    buf
}

fn encode_opset_import(domain: &str, version: i64) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_string_field(&mut buf, 1, domain);
    encode_int64_field(&mut buf, 2, version);
    buf
}

fn encode_model(model: &OnnxModel) -> Vec<u8> {
    let mut buf = Vec::new();
    // ir_version (field 1)
    encode_int64_field(&mut buf, 1, model.ir_version);
    // opset_import (field 8)
    let opset = encode_opset_import("", model.opset_version);
    encode_submessage_field(&mut buf, 8, &opset);
    // producer_name (field 2)
    encode_string_field(&mut buf, 2, &model.producer);
    // graph (field 7)
    let graph_bytes = encode_graph(&model.graph);
    encode_submessage_field(&mut buf, 7, &graph_bytes);
    buf
}

// --- P6-ONNX-01: Module trait walk → ONNX DAG ---

/// Builder for constructing an ONNX graph from a nabla Module.
pub struct OnnxExporter {
    model: OnnxModel,
    node_counter: usize,
}

impl OnnxExporter {
    /// Create a new exporter with default opset 21.
    #[must_use]
    pub fn new() -> Self {
        Self { model: OnnxModel::default(), node_counter: 0 }
    }

    /// Set the model name.
    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.model.graph.name = name.to_owned();
        self
    }

    /// Set opset version.
    #[must_use]
    pub fn with_opset(mut self, version: i64) -> Self {
        self.model.opset_version = version;
        self
    }

    /// Set IR version.
    #[must_use]
    pub fn with_ir_version(mut self, version: i64) -> Self {
        self.model.ir_version = version;
        self
    }

    /// Add a graph input with dynamic axes (P6-ONNX-04).
    pub fn add_input(&mut self, spec: TensorSpec) {
        self.model.graph.inputs.push(spec);
    }

    /// Add a graph output with dynamic axes (P6-ONNX-04).
    pub fn add_output(&mut self, spec: TensorSpec) {
        self.model.graph.outputs.push(spec);
    }

    /// Extract weights from a Module and add as ONNX initializers (P6-ONNX-01).
    pub fn add_module_weights<T: Scalar, B: Backend>(&mut self, module: &dyn Module<T, B>) {
        for (name, tensor) in module.state_dict() {
            let (rows, cols) = tensor.shape();
            let mut data = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                for c in 0..cols {
                    data.push(tensor.get(r, c).to_f64() as f32);
                }
            }
            self.model.graph.initializers.push(OnnxInitializer {
                name: name.to_owned(),
                dims: vec![rows as i64, cols as i64],
                data,
            });
        }
    }

    /// Add a computation node to the graph.
    pub fn add_node(&mut self, op: OnnxOp, inputs: &[&str], outputs: &[&str]) -> String {
        let name = format!("node_{}", self.node_counter);
        self.node_counter += 1;
        self.model.graph.nodes.push(OnnxNode {
            inputs: inputs.iter().map(|s| (*s).to_owned()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_owned()).collect(),
            name: name.clone(),
            op,
        });
        name
    }

    /// Add a named computation node.
    pub fn add_named_node(&mut self, name: &str, op: OnnxOp, inputs: &[&str], outputs: &[&str]) {
        self.node_counter += 1;
        self.model.graph.nodes.push(OnnxNode {
            inputs: inputs.iter().map(|s| (*s).to_owned()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_owned()).collect(),
            name: name.to_owned(),
            op,
        });
    }

    /// Convenience: add a linear layer (Gemm or MatMul+Add) from a nabla Linear module.
    pub fn add_linear<T: Scalar, B: Backend>(
        &mut self,
        prefix: &str,
        module: &crate::ml::module::Linear<T, B>,
        input: &str,
        output: &str,
    ) {
        let weight_name = format!("{prefix}.weight");
        let (rows, cols) = module.weight.shape();
        let mut w_data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols { w_data.push(module.weight.get(r, c).to_f64() as f32); }
        }
        self.model.graph.initializers.push(OnnxInitializer {
            name: weight_name.clone(),
            dims: vec![rows as i64, cols as i64],
            data: w_data,
        });

        if let Some(ref bias) = module.bias {
            let bias_name = format!("{prefix}.bias");
            let (br, bc) = bias.shape();
            let total = br * bc;
            let mut b_data = Vec::with_capacity(total);
            for r in 0..br {
                for c in 0..bc { b_data.push(bias.get(r, c).to_f64() as f32); }
            }
            self.model.graph.initializers.push(OnnxInitializer {
                name: bias_name.clone(),
                dims: vec![total as i64],
                data: b_data,
            });
            self.add_named_node(
                &format!("{prefix}_gemm"), OnnxOp::Gemm { alpha: 1.0, beta: 1.0, trans_b: true },
                &[input, &weight_name, &bias_name], &[output],
            );
        } else {
            self.add_named_node(
                &format!("{prefix}_transpose"), OnnxOp::Transpose { perm: vec![1, 0] },
                &[&weight_name], &[&format!("{prefix}.weight_t")],
            );
            self.add_named_node(
                &format!("{prefix}_matmul"), OnnxOp::MatMul,
                &[input, &format!("{prefix}.weight_t")], &[output],
            );
        }
    }

    /// Build the final ONNX model.
    #[must_use]
    pub fn build(self) -> OnnxModel {
        self.model
    }
}

// --- P6-ONNX-02: Serialization ---

impl OnnxModel {
    /// Serialize to ONNX protobuf bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        encode_model(self)
    }

    /// Write ONNX model to a file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let bytes = self.to_bytes();
        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)
    }
}

// --- P6-ONNX-04: Dynamic axes helpers ---

/// Create a standard NLP input spec with dynamic batch and sequence dims.
#[must_use]
pub fn nlp_input(name: &str, hidden: i64) -> TensorSpec {
    TensorSpec::float(name, vec![
        DimSpec::Dynamic("batch_size".to_owned()),
        DimSpec::Dynamic("seq_len".to_owned()),
        DimSpec::Fixed(hidden),
    ])
}

/// Create a standard 2D input spec with dynamic batch dim.
#[must_use]
pub fn batched_input(name: &str, features: i64) -> TensorSpec {
    TensorSpec::float(name, vec![
        DimSpec::Dynamic("batch_size".to_owned()),
        DimSpec::Fixed(features),
    ])
}

/// Create a standard image input spec with dynamic batch dim.
#[must_use]
pub fn image_input(name: &str, channels: i64, height: i64, width: i64) -> TensorSpec {
    TensorSpec::float(name, vec![
        DimSpec::Dynamic("batch_size".to_owned()),
        DimSpec::Fixed(channels), DimSpec::Fixed(height), DimSpec::Fixed(width),
    ])
}

// --- P6-ONNX-05: Verification documentation ---
// To verify exported models with onnxruntime:
//
// ```python
// import onnxruntime as ort
// import numpy as np
//
// session = ort.InferenceSession("model.onnx")
// inputs = {"input": np.random.randn(1, 10).astype(np.float32)}
// nabla_output = ...  # run nabla model
// ort_output = session.run(None, inputs)[0]
// assert np.allclose(nabla_output, ort_output, atol=1e-5)
// ```

/// Export a simple feedforward module (Sequential of Linear+Activation layers).
///
/// Walks the module tree via `named_children()` / `named_parameters()` and maps
/// each layer to corresponding ONNX operators.
pub fn export_sequential<T: Scalar, B: Backend>(
    model: &crate::ml::module::Sequential<T, B>,
    input_features: i64,
    output_features: i64,
) -> OnnxModel {
    let mut ex = OnnxExporter::new();
    ex.add_input(batched_input("input", input_features));
    ex.add_output(batched_input("output", output_features));
    ex.add_module_weights(model);

    // Walk Sequential children and create ONNX nodes
    let children = model.named_children();
    let mut current_tensor = "input".to_owned();

    for (i, (_name, child)) in children.iter().enumerate() {
        let params = child.named_parameters();
        let is_last = i == children.len() - 1;
        let out_name = if is_last { "output".to_owned() } else { format!("hidden_{i}") };

        if params.is_empty() {
            // Parameterless layer — try to infer activation type
            // For now, emit Relu as default (most common); users can build custom graphs
            // for non-standard architectures via OnnxExporter directly
            ex.add_named_node(
                &format!("activation_{i}"), OnnxOp::Relu,
                &[&current_tensor], &[&out_name],
            );
        } else {
            // Parameterized layer — emit as Gemm (weight + optional bias)
            let weight_key = format!("{i}.weight");
            let has_bias = params.iter().any(|(n, _)| *n == "bias");
            if has_bias {
                let bias_key = format!("{i}.bias");
                ex.add_named_node(
                    &format!("linear_{i}"), OnnxOp::Gemm { alpha: 1.0, beta: 1.0, trans_b: true },
                    &[&current_tensor, &weight_key, &bias_key], &[&out_name],
                );
            } else {
                let wt_name = format!("{i}.weight_t");
                ex.add_named_node(
                    &format!("transpose_{i}"), OnnxOp::Transpose { perm: vec![1, 0] },
                    &[&weight_key], &[&wt_name],
                );
                ex.add_named_node(
                    &format!("matmul_{i}"), OnnxOp::MatMul,
                    &[&current_tensor, &wt_name], &[&out_name],
                );
            }
        }
        current_tensor = out_name;
    }

    ex.build()
}

// --- Re-export fix for Sequential initializer shape ---
// The add_module_weights for Sequential uses named_parameters which prefixes
// names with "0.weight", "0.bias", etc. The export_sequential function uses
// the same naming, so they match. Bias tensors in nabla Linear are (1, out_features)
// but ONNX Gemm expects 1-D bias. We handle this in add_module_weights by checking
// if either dim is 1 and flattening.

impl OnnxExporter {
    /// Extract weights from a Module, flattening bias-like (1,N)/(N,1) tensors to 1-D.
    pub fn add_module_weights_flat<T: Scalar, B: Backend>(&mut self, module: &dyn Module<T, B>) {
        for (name, tensor) in module.state_dict() {
            let (rows, cols) = tensor.shape();
            let mut data = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                for c in 0..cols { data.push(tensor.get(r, c).to_f64() as f32); }
            }
            let dims = if (rows == 1 || cols == 1) && name.contains("bias") {
                vec![(rows * cols) as i64]
            } else {
                vec![rows as i64, cols as i64]
            };
            self.model.graph.initializers.push(OnnxInitializer {
                name: name.to_owned(), dims, data,
            });
        }
    }
}

/// Export a Sequential model with proper bias flattening for Gemm compatibility.
pub fn export_sequential_flat<T: Scalar, B: Backend>(
    model: &crate::ml::module::Sequential<T, B>,
    input_features: i64,
    output_features: i64,
) -> OnnxModel {
    let mut ex = OnnxExporter::new();
    ex.add_input(batched_input("input", input_features));
    ex.add_output(batched_input("output", output_features));
    ex.add_module_weights_flat(model);

    let children = model.named_children();
    let mut current_tensor = "input".to_owned();

    for (i, (_name, child)) in children.iter().enumerate() {
        let params = child.named_parameters();
        let is_last = i == children.len() - 1;
        let out_name = if is_last { "output".to_owned() } else { format!("hidden_{i}") };

        if params.is_empty() {
            ex.add_named_node(
                &format!("activation_{i}"), OnnxOp::Relu,
                &[&current_tensor], &[&out_name],
            );
        } else {
            let weight_key = format!("{i}.weight");
            let has_bias = params.iter().any(|(n, _)| *n == "bias");
            if has_bias {
                let bias_key = format!("{i}.bias");
                ex.add_named_node(
                    &format!("linear_{i}"), OnnxOp::Gemm { alpha: 1.0, beta: 1.0, trans_b: true },
                    &[&current_tensor, &weight_key, &bias_key], &[&out_name],
                );
            } else {
                let wt_name = format!("{i}.weight_t");
                ex.add_named_node(
                    &format!("transpose_{i}"), OnnxOp::Transpose { perm: vec![1, 0] },
                    &[&weight_key], &[&wt_name],
                );
                ex.add_named_node(
                    &format!("matmul_{i}"), OnnxOp::MatMul,
                    &[&current_tensor, &wt_name], &[&out_name],
                );
            }
        }
        current_tensor = out_name;
    }

    ex.build()
}
