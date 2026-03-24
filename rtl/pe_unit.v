`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : pe_unit
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Processing element implementing local multiply-accumulate operation.

Maintains an output-stationary partial sum while forwarding operands to
neighboring PEs for continued computation across the array.

Key Points
----------
- Performs signed MAC using reusable mac_unit
- Supports clear and seeded accumulation control
- Keeps accumulation local (output-stationary dataflow)
- Forwards activation, weight, and valid signals
------------------------------------------------------------------------------
*/

module pe_unit #(
    parameter integer DATA_WIDTH = 16,
    parameter integer ACC_WIDTH  = 40
)(
    input  wire                            clk,
    input  wire                            rst,

    // Data inputs to this PE
    input  wire signed [DATA_WIDTH-1:0]    act_in,
    input  wire signed [DATA_WIDTH-1:0]    weight_in,
    input  wire                            valid_in,

    // Local psum control
    input  wire                            clear_psum,
    input  wire                            load_psum,
    input  wire signed [ACC_WIDTH-1:0]     psum_seed,

    // Forwarded outputs to neighboring PEs
    output reg  signed [DATA_WIDTH-1:0]    act_out,
    output reg  signed [DATA_WIDTH-1:0]    weight_out,
    output reg                             valid_out,

    // Local accumulated partial sum
    output wire signed [ACC_WIDTH-1:0]     psum_out
);

    // ------------------------------------------------------------------------
    // Safety check
    // ------------------------------------------------------------------------
    initial begin
        if (ACC_WIDTH < (2 * DATA_WIDTH)) begin
            $error("pe_unit: ACC_WIDTH must be >= 2*DATA_WIDTH");
            $finish;
        end
    end

    // ------------------------------------------------------------------------
    // Internal MAC wiring
    // mac_unit's registered output is the PE's stationary psum state
    // ------------------------------------------------------------------------
    wire signed [ACC_WIDTH-1:0] current_psum;
    wire signed [ACC_WIDTH-1:0] mac_acc_in;
    wire signed [DATA_WIDTH-1:0] act_mux;
    wire signed [DATA_WIDTH-1:0] weight_mux;
    wire                         mac_en;

    // MAC runs when:
    // - valid data is present, or
    // - psum needs to be cleared, or
    // - psum needs to be loaded from seed
    assign mac_en = valid_in | clear_psum | load_psum;

    // On control-only cycles, force multiply term to zero
    assign act_mux    = valid_in ? act_in    : {DATA_WIDTH{1'b0}};
    assign weight_mux = valid_in ? weight_in : {DATA_WIDTH{1'b0}};

    // Accumulator source selection
    // clear has highest priority
    assign mac_acc_in = clear_psum ? {ACC_WIDTH{1'b0}} :
                        load_psum  ? psum_seed          :
                                     current_psum;

    mac_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) u_mac_unit (
        .clk         (clk),
        .rst         (rst),
        .en          (mac_en),
        .input_data  (act_mux),
        .weight_data (weight_mux),
        .acc_in      (mac_acc_in),
        .acc_out     (current_psum)
    );

    // ------------------------------------------------------------------------
    // Forward activation / weight / valid to neighboring PE(s)
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            act_out    <= {DATA_WIDTH{1'b0}};
            weight_out <= {DATA_WIDTH{1'b0}};
            valid_out  <= 1'b0;
        end
        else begin
            act_out    <= act_in;
            weight_out <= weight_in;
            valid_out  <= valid_in;
        end
    end

    assign psum_out = current_psum;

endmodule