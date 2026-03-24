`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : pe_array_4x4
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
4x4 processing-element mesh forming the core compute fabric of NNIA.

Enables parallel MAC operations using structured dataflow where activations
propagate horizontally and weights propagate vertically.

Key Points
----------
- Central tiled compute engine of NNIA
- Activation flow: left → right, Weight flow: top → bottom
- MAC executes only when both operand valids align
- Outputs per-PE partial sums for tiled accumulation flow
------------------------------------------------------------------------------
*/

module pe_array_4x4 #(
    parameter integer ROWS       = 4,
    parameter integer COLS       = 4,
    parameter integer DATA_WIDTH = 16,
    parameter integer ACC_WIDTH  = 40
)(
    input  wire                                      clk,
    input  wire                                      rst,

    // ------------------------------------------------------------------------
    // Left-edge activation inputs: one activation per row
    // row_act_in[row]
    // ------------------------------------------------------------------------
    input  wire signed [(ROWS*DATA_WIDTH)-1:0]       row_act_in,

    // ------------------------------------------------------------------------
    // Top-edge weight inputs: one weight per column
    // col_weight_in[col]
    // ------------------------------------------------------------------------
    input  wire signed [(COLS*DATA_WIDTH)-1:0]       col_weight_in,

    // ------------------------------------------------------------------------
    // Left-edge activation-valid inputs: one per row
    // row_act_valid_in[row]
    // ------------------------------------------------------------------------
    input  wire        [ROWS-1:0]                    row_act_valid_in,

    // ------------------------------------------------------------------------
    // Top-edge weight-valid inputs: one per column
    // col_weight_valid_in[col]
    // ------------------------------------------------------------------------
    input  wire        [COLS-1:0]                    col_weight_valid_in,

    // ------------------------------------------------------------------------
    // Broadcast control for PE local psum state
    // ------------------------------------------------------------------------
    input  wire                                      clear_psum,
    input  wire                                      load_psum,

    // ------------------------------------------------------------------------
    // One psum seed per PE, flattened row-major
    // ------------------------------------------------------------------------
    input  wire signed [(ROWS*COLS*ACC_WIDTH)-1:0]   psum_seed_flat,

    // ------------------------------------------------------------------------
    // Boundary / observation outputs
    // ------------------------------------------------------------------------
    output wire signed [(ROWS*DATA_WIDTH)-1:0]       right_edge_act_out,
    output wire signed [(COLS*DATA_WIDTH)-1:0]       bottom_edge_weight_out,
    output wire        [ROWS-1:0]                    right_edge_act_valid_out,
    output wire        [COLS-1:0]                    bottom_edge_weight_valid_out,

    // ------------------------------------------------------------------------
    // One psum output per PE, flattened row-major
    // ------------------------------------------------------------------------
    output wire signed [(ROWS*COLS*ACC_WIDTH)-1:0]   psum_out_flat
);

    // ------------------------------------------------------------------------
    // Safety checks
    // ------------------------------------------------------------------------
    initial begin
        if (ROWS != 4 || COLS != 4) begin
            $error("pe_array_4x4: This module is intended for a 4x4 PE array.");
            $finish;
        end

        if (ACC_WIDTH < (2 * DATA_WIDTH)) begin
            $error("pe_array_4x4: ACC_WIDTH must be >= 2*DATA_WIDTH");
            $finish;
        end
    end

    // ------------------------------------------------------------------------
    // Internal forwarded data buses from each PE
    // ------------------------------------------------------------------------
    wire signed [(ROWS*COLS*DATA_WIDTH)-1:0] act_fwd_bus;
    wire signed [(ROWS*COLS*DATA_WIDTH)-1:0] weight_fwd_bus;

    // Separate propagated valid buses
    wire        [(ROWS*COLS)-1:0]            act_valid_fwd_bus;
    wire        [(ROWS*COLS)-1:0]            weight_valid_fwd_bus;

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : GEN_ROWS
            for (c = 0; c < COLS; c = c + 1) begin : GEN_COLS

                localparam integer PE_INDEX = (r * COLS) + c;

                wire signed [DATA_WIDTH-1:0] pe_act_in;
                wire signed [DATA_WIDTH-1:0] pe_weight_in;
                wire                         pe_act_valid_in;
                wire                         pe_weight_valid_in;
                wire                         pe_valid_in;
                wire signed [ACC_WIDTH-1:0]  pe_psum_seed;
                wire                         pe_valid_out_unused;

                // ------------------------------------------------------------
                // Activation path:
                // first column gets external row input,
                // otherwise activation comes from left PE
                // ------------------------------------------------------------
                if (c == 0) begin : GEN_ACT_LEFT_EDGE
                    assign pe_act_in       = row_act_in[(r*DATA_WIDTH) +: DATA_WIDTH];
                    assign pe_act_valid_in = row_act_valid_in[r];
                end
                else begin : GEN_ACT_FROM_LEFT
                    localparam integer LEFT_INDEX = (r * COLS) + (c - 1);
                    assign pe_act_in       = act_fwd_bus[(LEFT_INDEX*DATA_WIDTH) +: DATA_WIDTH];
                    assign pe_act_valid_in = act_valid_fwd_bus[LEFT_INDEX];
                end

                // ------------------------------------------------------------
                // Weight path:
                // first row gets external column input,
                // otherwise weight comes from upper PE
                // ------------------------------------------------------------
                if (r == 0) begin : GEN_WEIGHT_TOP_EDGE
                    assign pe_weight_in       = col_weight_in[(c*DATA_WIDTH) +: DATA_WIDTH];
                    assign pe_weight_valid_in = col_weight_valid_in[c];
                end
                else begin : GEN_WEIGHT_FROM_UP
                    localparam integer UP_INDEX = ((r - 1) * COLS) + c;
                    assign pe_weight_in       = weight_fwd_bus[(UP_INDEX*DATA_WIDTH) +: DATA_WIDTH];
                    assign pe_weight_valid_in = weight_valid_fwd_bus[UP_INDEX];
                end

                // ------------------------------------------------------------
                // MAC enable for this PE
                // ------------------------------------------------------------
                assign pe_valid_in  = pe_act_valid_in & pe_weight_valid_in;
                assign pe_psum_seed = psum_seed_flat[(PE_INDEX*ACC_WIDTH) +: ACC_WIDTH];

                // ------------------------------------------------------------
                // PE instance
                // ------------------------------------------------------------
                pe_unit #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH (ACC_WIDTH)
                ) u_pe_unit (
                    .clk        (clk),
                    .rst        (rst),

                    .act_in     (pe_act_in),
                    .weight_in  (pe_weight_in),
                    .valid_in   (pe_valid_in),

                    .clear_psum (clear_psum),
                    .load_psum  (load_psum),
                    .psum_seed  (pe_psum_seed),

                    .act_out    (act_fwd_bus[(PE_INDEX*DATA_WIDTH) +: DATA_WIDTH]),
                    .weight_out (weight_fwd_bus[(PE_INDEX*DATA_WIDTH) +: DATA_WIDTH]),
                    .valid_out  (pe_valid_out_unused),

                    .psum_out   (psum_out_flat[(PE_INDEX*ACC_WIDTH) +: ACC_WIDTH])
                );

                // ------------------------------------------------------------
                // Separate valid propagation aligned with forwarded data
                // act-valid moves horizontally
                // weight-valid moves vertically
                // ------------------------------------------------------------
                reg act_valid_reg;
                reg weight_valid_reg;

                always @(posedge clk) begin
                    if (rst) begin
                        act_valid_reg    <= 1'b0;
                        weight_valid_reg <= 1'b0;
                    end
                    else begin
                        act_valid_reg    <= pe_act_valid_in;
                        weight_valid_reg <= pe_weight_valid_in;
                    end
                end

                assign act_valid_fwd_bus[PE_INDEX]    = act_valid_reg;
                assign weight_valid_fwd_bus[PE_INDEX] = weight_valid_reg;

            end
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Boundary observation outputs
    // ------------------------------------------------------------------------
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : GEN_RIGHT_EDGE
            localparam integer RIGHT_INDEX = (r * COLS) + (COLS - 1);

            assign right_edge_act_out[(r*DATA_WIDTH) +: DATA_WIDTH] =
                act_fwd_bus[(RIGHT_INDEX*DATA_WIDTH) +: DATA_WIDTH];

            assign right_edge_act_valid_out[r] =
                act_valid_fwd_bus[RIGHT_INDEX];
        end

        for (c = 0; c < COLS; c = c + 1) begin : GEN_BOTTOM_EDGE
            localparam integer BOTTOM_INDEX = ((ROWS - 1) * COLS) + c;

            assign bottom_edge_weight_out[(c*DATA_WIDTH) +: DATA_WIDTH] =
                weight_fwd_bus[(BOTTOM_INDEX*DATA_WIDTH) +: DATA_WIDTH];

            assign bottom_edge_weight_valid_out[c] =
                weight_valid_fwd_bus[BOTTOM_INDEX];
        end
    endgenerate

endmodule