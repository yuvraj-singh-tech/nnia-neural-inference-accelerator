`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : tile_controller
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Central control FSM for tiled NNIA execution.

This controller coordinates input streaming, weight streaming, accumulation,
and output writeback across tiled matrix operations. It enforces a deterministic
execution order while ensuring correctness through a bounded pipeline drain.

Design Characteristics
----------------------
- Sequential tile execution with strict data dependency control
- Latency-aware drain phase to guarantee complete pipeline flush
- Clean orchestration of buffers, PE array, and output stages
- Stable and deterministic behavior aligned with simulation correctness

Execution Flow
--------------
1. Start run and initialize output buffer
2. For each output tile:
   - For each K tile:
       • Start input and weight streaming
       • Perform accumulation across TILE_K cycles
       • Apply controlled drain to flush pipeline
       • Save accumulated partial sums
   - After final K tile:
       • Write output tile
3. Advance to next output tile
4. Assert done when all tiles complete

------------------------------------------------------------------------------
*/

module tile_controller #(
    parameter integer TILE_K = 4,
    parameter integer ROWS   = 4,
    parameter integer COLS   = 4
)(
    input  wire clk,
    input  wire rst,

    input  wire start,
    output reg  busy,
    output reg  done,

    output reg  addr_start_run,
    output reg  addr_advance_k,
    output reg  addr_advance_out,

    input  wire first_k_tile,
    input  wire last_k_tile,
    input  wire first_output_tile,
    input  wire last_output_tile,
    input  wire addr_active,
    input  wire addr_done,
    input  wire cmd_error,

    output reg  inbuf_start,
    output reg  inbuf_advance,
    input  wire inbuf_tile_done,
    input  wire inbuf_stream_active,
    input  wire inbuf_cfg_error,

    output reg  wbuf_start,
    output reg  wbuf_advance,
    input  wire wbuf_tile_done,
    input  wire wbuf_stream_active,
    input  wire wbuf_cfg_error,

    output reg  psum_clear,
    output reg  psum_save,

    output reg  pe_clear_psum,
    output reg  pe_load_psum,

    output reg  outbuf_clear,
    output reg  outbuf_save,

    output reg  ctrl_error
);

    // ------------------------------------------------------------------------
    // FSM States
    // ------------------------------------------------------------------------
    localparam [3:0]
        S_IDLE         = 4'd0,
        S_START_RUN    = 4'd1,
        S_START_STREAM = 4'd2,
        S_STREAM       = 4'd3,
        S_DRAIN        = 4'd4,
        S_SAVE_PSUM    = 4'd5,
        S_SAVE_OUTPUT  = 4'd6,
        S_ADVANCE_OUT  = 4'd7,
        S_DONE         = 4'd8,
        S_ERROR        = 4'd9;

    reg [3:0] state, next_state;

    // ------------------------------------------------------------------------
    // Counters
    // ------------------------------------------------------------------------
    localparam integer STREAM_CNT_W =
        (TILE_K <= 1) ? 1 : $clog2(TILE_K + 1);

    reg [STREAM_CNT_W-1:0] stream_count;
    reg [2:0] drain_count;

    // ------------------------------------------------------------------------
    // Start edge detection
    // ------------------------------------------------------------------------
    reg start_d;
    wire start_rise;

    always @(posedge clk) begin
        if (rst) start_d <= 1'b0;
        else     start_d <= start;
    end

    assign start_rise = start & ~start_d;

    // ------------------------------------------------------------------------
    // State register
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) state <= S_IDLE;
        else     state <= next_state;
    end

    // ------------------------------------------------------------------------
    // Stream counter (K-dimension accumulation)
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst)
            stream_count <= 0;
        else begin
            case (state)
                S_START_STREAM:
                    stream_count <= 0;

                S_STREAM:
                    if (stream_count < TILE_K)
                        stream_count <= stream_count + 1'b1;
            endcase
        end
    end

    // ------------------------------------------------------------------------
    // Drain counter (pipeline flush window)
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst)
            drain_count <= 0;
        else if (state == S_DRAIN)
            drain_count <= drain_count + 1'b1;
        else
            drain_count <= 0;
    end

    // ------------------------------------------------------------------------
    // Next-state logic
    // ------------------------------------------------------------------------
    always @(*) begin
        next_state = state;

        case (state)

            S_IDLE:
                if (start_rise)
                    next_state = S_START_RUN;

            S_START_RUN:
                next_state = S_START_STREAM;

            S_START_STREAM:
                next_state = S_STREAM;

            S_STREAM: begin
                if (cmd_error || inbuf_cfg_error || wbuf_cfg_error)
                    next_state = S_ERROR;
                else if (stream_count == TILE_K)
                    next_state = S_DRAIN;
            end

            S_DRAIN:
                if (drain_count == 5)
                    next_state = S_SAVE_PSUM;

            S_SAVE_PSUM:
                if (last_k_tile)
                    next_state = S_SAVE_OUTPUT;
                else
                    next_state = S_START_STREAM;

            S_SAVE_OUTPUT:
                next_state = S_ADVANCE_OUT;

            S_ADVANCE_OUT:
                if (last_output_tile)
                    next_state = S_DONE;
                else
                    next_state = S_START_STREAM;

            S_DONE:
                if (!start)
                    next_state = S_IDLE;

            S_ERROR:
                next_state = S_ERROR;

            default:
                next_state = S_ERROR;
        endcase
    end

    // ------------------------------------------------------------------------
    // Output and control signal generation
    // ------------------------------------------------------------------------
    always @(*) begin
        addr_start_run   = 0;
        addr_advance_k   = 0;
        addr_advance_out = 0;

        inbuf_start   = 0;
        inbuf_advance = 0;

        wbuf_start   = 0;
        wbuf_advance = 0;

        psum_clear = 0;
        psum_save  = 0;

        pe_clear_psum = 0;
        pe_load_psum  = 0;

        outbuf_clear = 0;
        outbuf_save  = 0;

        busy       = 1;
        done       = 0;
        ctrl_error = 0;

        case (state)

            S_IDLE:
                busy = 0;

            S_START_RUN: begin
                addr_start_run = 1;
                outbuf_clear   = 1;
            end

            S_START_STREAM: begin
                inbuf_start = 1;
                wbuf_start  = 1;

                if (first_k_tile) begin
                    psum_clear    = 1;
                    pe_clear_psum = 1;
                end
                else begin
                    pe_load_psum = 1;
                end
            end

            S_STREAM: begin
                if (stream_count < TILE_K) begin
                    inbuf_advance = 1;
                    wbuf_advance  = 1;
                end
            end

            S_SAVE_PSUM: begin
                psum_save = 1;
                if (!last_k_tile)
                    addr_advance_k = 1;
            end

            S_SAVE_OUTPUT:
                outbuf_save = 1;

            S_ADVANCE_OUT:
                addr_advance_out = 1;

            S_DONE: begin
                busy = 0;
                done = 1;
            end

            S_ERROR: begin
                busy       = 0;
                ctrl_error = 1;
            end
        endcase
    end

endmodule
