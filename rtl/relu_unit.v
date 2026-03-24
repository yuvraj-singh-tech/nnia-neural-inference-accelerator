`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : relu_unit
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Combinational activation block implementing ReLU.

Applies element-wise non-linearity by suppressing negative values and
passing non-negative values unchanged, aligning with standard neural
inference behavior.

Key Points
----------
- Pure combinational logic (no clock/reset)
- Operates directly on signed DATA_WIDTH input
- Used after quantization and bias addition stages
- No scaling, clipping, or additional processing
------------------------------------------------------------------------------
*/

module relu_unit #(
    parameter integer DATA_WIDTH = 16
)(
    input  wire signed [DATA_WIDTH-1:0] input_data,
    output wire signed [DATA_WIDTH-1:0] output_data
);

    assign output_data = input_data[DATA_WIDTH-1] ? {DATA_WIDTH{1'b0}} : input_data;

endmodule