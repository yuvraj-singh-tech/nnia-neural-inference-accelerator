`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : quant_bias_relu
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Post-processing stage converting raw accumulator output to final result.

Applies fixed-point requantization, bias addition, activation, and range
control while preserving alignment with the reference software model.

Key Points
----------
- Processing order: requantize → bias → clamp → ReLU → final clamp
- Operates from ACC_WIDTH domain to DATA_WIDTH output
- Ensures numerical stability and bounded outputs
- Maintains hardware-software consistency
------------------------------------------------------------------------------
*/

module quant_bias_relu #(
    parameter integer DATA_WIDTH = 16,
    parameter integer FRAC_BITS  = 8,
    parameter integer ACC_WIDTH  = 40
)(
    input  wire signed [ACC_WIDTH-1:0]  raw_acc_in,
    input  wire signed [DATA_WIDTH-1:0] bias_data,
    output wire signed [DATA_WIDTH-1:0] out_data
);

    // ------------------------------------------------------------------------
    // Safety checks
    // ------------------------------------------------------------------------
    initial begin
        if (ACC_WIDTH < DATA_WIDTH) begin
            $error("quant_bias_relu: ACC_WIDTH must be >= DATA_WIDTH");
            $finish;
        end
    end

    // ------------------------------------------------------------------------
    // Constants
    // ------------------------------------------------------------------------
    localparam signed [ACC_WIDTH-1:0] ACC_MAX_VAL = {1'b0, {(ACC_WIDTH-1){1'b1}}};
    localparam signed [ACC_WIDTH-1:0] ACC_MIN_VAL = {1'b1, {(ACC_WIDTH-1){1'b0}}};

    localparam signed [DATA_WIDTH-1:0] DATA_MAX_VAL = {1'b0, {(DATA_WIDTH-1){1'b1}}};
    localparam signed [DATA_WIDTH-1:0] DATA_MIN_VAL = {1'b1, {(DATA_WIDTH-1){1'b0}}};

    // Widened clamp bounds for ACC-domain comparison
    localparam signed [ACC_WIDTH:0] ACC_MAX_EXT = {1'b0, ACC_MAX_VAL};
    localparam signed [ACC_WIDTH:0] ACC_MIN_EXT = {1'b1, ACC_MIN_VAL};

    // ------------------------------------------------------------------------
    // Datapath
    // ------------------------------------------------------------------------
    wire signed [ACC_WIDTH-1:0] shifted_acc;
    wire signed [ACC_WIDTH-1:0] bias_ext_narrow;

    wire signed [ACC_WIDTH:0]   shifted_acc_ext;
    wire signed [ACC_WIDTH:0]   bias_ext_wide;
    wire signed [ACC_WIDTH:0]   biased_sum_ext;

    wire signed [ACC_WIDTH-1:0] acc_clipped;
    wire signed [ACC_WIDTH-1:0] acc_after_relu;

    wire signed [ACC_WIDTH-1:0] data_max_ext;
    wire signed [ACC_WIDTH-1:0] data_min_ext;

    // Step 1: arithmetic requantization
    assign shifted_acc = raw_acc_in >>> FRAC_BITS;

    // Step 2: sign-extend bias
    assign bias_ext_narrow = {{(ACC_WIDTH-DATA_WIDTH){bias_data[DATA_WIDTH-1]}}, bias_data};

    // Step 2b: widen both operands for safe addition/comparison
    assign shifted_acc_ext = {shifted_acc[ACC_WIDTH-1], shifted_acc};
    assign bias_ext_wide   = {{(ACC_WIDTH+1-DATA_WIDTH){bias_data[DATA_WIDTH-1]}}, bias_data};

    // Step 2c: add in widened domain
    assign biased_sum_ext = shifted_acc_ext + bias_ext_wide;

    // Step 3: clamp to ACC_WIDTH
    assign acc_clipped =
        (biased_sum_ext > ACC_MAX_EXT) ? ACC_MAX_VAL :
        (biased_sum_ext < ACC_MIN_EXT) ? ACC_MIN_VAL :
                                         biased_sum_ext[ACC_WIDTH-1:0];

    // Step 4: ReLU in ACC domain
    relu_unit #(
        .DATA_WIDTH(ACC_WIDTH)
    ) u_relu_unit_acc (
        .input_data  (acc_clipped),
        .output_data (acc_after_relu)
    );

    // Step 5: final clamp to DATA_WIDTH
    assign data_max_ext = {{(ACC_WIDTH-DATA_WIDTH){1'b0}}, DATA_MAX_VAL};
    assign data_min_ext = {{(ACC_WIDTH-DATA_WIDTH){1'b1}}, DATA_MIN_VAL};

    assign out_data =
        (acc_after_relu > data_max_ext) ? DATA_MAX_VAL :
        (acc_after_relu < data_min_ext) ? DATA_MIN_VAL :
                                          acc_after_relu[DATA_WIDTH-1:0];

endmodule