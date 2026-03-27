`timescale 1ns / 1ps

/*
------------------------------------------------------------------------------
Module Name : tb_top_nnia
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Full-system verification testbench for NNIA.

Executes complete inference flow using Python-generated inputs and validates
RTL outputs against the reference model, ensuring functional correctness of
the tiled accelerator.

Key Points
----------
- Loads input, weight, bias, and expected output from .mem files
- Drives full NNIA pipeline through top-level interface
- Includes watchdog and timeout protection for safe execution
- Captures RTL outputs and writes rtl_output.mem for external comparison
- Provides detailed match/mismatch reporting

Role in Flow
------------
- Acts as simulation bridge between Python model and RTL accelerator
- Validates end-to-end correctness of tiled NNIA execution
- Enables performance extraction and debugging visibility
------------------------------------------------------------------------------
*/

module tb_top_nnia;

    // ------------------------------------------------------------------------
    // Locked NNIA configuration
    // ------------------------------------------------------------------------
    localparam integer M_TOTAL    = 4;
    localparam integer K_TOTAL    = 16;
    localparam integer N_TOTAL    = 8;
    localparam integer TILE_M     = 4;
    localparam integer TILE_K     = 4;
    localparam integer TILE_N     = 4;
    localparam integer ROWS       = 4;
    localparam integer COLS       = 4;
    localparam integer DATA_WIDTH = 16;
    localparam integer FRAC_BITS  = 8;
    localparam integer ACC_WIDTH  = 40;

    localparam integer INPUT_ELEMS   = M_TOTAL * K_TOTAL;
    localparam integer WEIGHT_ELEMS  = K_TOTAL * N_TOTAL;
    localparam integer BIAS_ELEMS    = N_TOTAL;
    localparam integer OUTPUT_ELEMS  = M_TOTAL * N_TOTAL;

    localparam integer CLK_PERIOD_NS   = 10;
    localparam integer MAX_CYCLES      = 3000;
    localparam integer MAX_OUTPUT_WAIT = 100;

    parameter MEM_BASE_PATH = "mem/";

    // ------------------------------------------------------------------------
    // DUT interface
    // ------------------------------------------------------------------------
    reg                                            clk;
    reg                                            rst;
    reg                                            start;

    reg  signed [(M_TOTAL*K_TOTAL*DATA_WIDTH)-1:0] input_flat;
    reg  signed [(K_TOTAL*N_TOTAL*DATA_WIDTH)-1:0] weight_flat;
    reg  signed [(N_TOTAL*DATA_WIDTH)-1:0]         bias_flat;

    wire                                           busy;
    wire                                           done;
    wire                                           error;
    wire signed [(M_TOTAL*N_TOTAL*DATA_WIDTH)-1:0] final_output_flat;
    wire                                           output_valid;

    // ------------------------------------------------------------------------
    // Performance counter outputs from DUT
    // ------------------------------------------------------------------------
    wire [31:0] perf_total_cycles;
    wire [31:0] perf_run_cycles;
    wire [31:0] perf_run_count;
    wire [31:0] perf_tile_writes;
    wire [31:0] perf_psum_saves;

    // ------------------------------------------------------------------------
    // Software-side memories
    // ------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] input_mem    [0:INPUT_ELEMS-1];
    reg signed [DATA_WIDTH-1:0] weight_mem   [0:WEIGHT_ELEMS-1];
    reg signed [DATA_WIDTH-1:0] bias_mem     [0:BIAS_ELEMS-1];
    reg signed [DATA_WIDTH-1:0] expected_mem [0:OUTPUT_ELEMS-1];
    reg signed [DATA_WIDTH-1:0] rtl_out_mem  [0:OUTPUT_ELEMS-1];

    // ------------------------------------------------------------------------
    // Bookkeeping
    // ------------------------------------------------------------------------
    integer i;
    integer fd;
    integer cycle_count;
    integer mismatch_count;
    integer match_count;
    integer watchdog_cycles;
    integer wd_count;
    integer raw_psum_idx;
    integer dbg_c;

    reg     error_seen;
    reg     done_seen;
    reg [DATA_WIDTH-1:0] raw_bits;

    reg signed [DATA_WIDTH-1:0] actual_val;
    reg signed [DATA_WIDTH-1:0] expected_val;

    // ------------------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------------------
    top_nnia #(
        .M_TOTAL    (M_TOTAL),
        .K_TOTAL    (K_TOTAL),
        .N_TOTAL    (N_TOTAL),
        .TILE_M     (TILE_M),
        .TILE_K     (TILE_K),
        .TILE_N     (TILE_N),
        .ROWS       (ROWS),
        .COLS       (COLS),
        .DATA_WIDTH (DATA_WIDTH),
        .FRAC_BITS  (FRAC_BITS),
        .ACC_WIDTH  (ACC_WIDTH)
    ) dut (
        .clk               (clk),
        .rst               (rst),
        .start             (start),
        .input_flat        (input_flat),
        .weight_flat       (weight_flat),
        .bias_flat         (bias_flat),
        .busy              (busy),
        .done              (done),
        .error             (error),
        .final_output_flat (final_output_flat),
        .output_valid      (output_valid),

        .perf_total_cycles (perf_total_cycles),
        .perf_run_cycles   (perf_run_cycles),
        .perf_run_count    (perf_run_count),
        .perf_tile_writes  (perf_tile_writes),
        .perf_psum_saves   (perf_psum_saves)
    );

    // ------------------------------------------------------------------------
    // Clock
    // ------------------------------------------------------------------------
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD_NS/2) clk = ~clk;
    end

    // ------------------------------------------------------------------------
    // Runtime counters / status flags
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            cycle_count     <= 0;
            watchdog_cycles <= 0;
            error_seen      <= 1'b0;
            done_seen       <= 1'b0;
        end
        else begin
            cycle_count     <= cycle_count + 1;
            watchdog_cycles <= watchdog_cycles + 1;

            if (error)
                error_seen <= 1'b1;

            if (done)
                done_seen <= 1'b1;
        end
    end

    // ------------------------------------------------------------------------
    // Debug trigger:
    // Print useful row0 tile data exactly at writeback point
    // ------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst && dut.outbuf_save_d) begin
            $display("================================================================");
            $display("[NNIA TB][DEBUG] Tile writeback checkpoint | cycle=%0d", cycle_count);
            $display("  saved_tile_n_base_d      = %0d", dut.saved_tile_n_base_d);
            $display("  saved_last_output_tile_d = %0b", dut.saved_last_output_tile_d);

            $display("[NNIA TB][DEBUG] Row0 raw PE partial sums:");
            for (dbg_c = 0; dbg_c < COLS; dbg_c = dbg_c + 1) begin
                raw_psum_idx = (0 * COLS) + dbg_c;
                $display("  pe_psum_row0[%0d]        = %0d",
                         dbg_c,
                         $signed(dut.pe_psum_out_flat[(raw_psum_idx*ACC_WIDTH) +: ACC_WIDTH]));
            end

            $display("[NNIA TB][DEBUG] Row0 psum buffer contents:");
            for (dbg_c = 0; dbg_c < COLS; dbg_c = dbg_c + 1) begin
                raw_psum_idx = (0 * COLS) + dbg_c;
                $display("  psum_buf_row0[%0d]       = %0d",
                         dbg_c,
                         $signed(dut.psum_buf_out_flat[(raw_psum_idx*ACC_WIDTH) +: ACC_WIDTH]));
            end

            $display("[NNIA TB][DEBUG] Row0 processed tile values:");
            for (dbg_c = 0; dbg_c < COLS; dbg_c = dbg_c + 1) begin
                $display("  processed_row0[%0d]      = %0d",
                         dbg_c,
                         $signed(dut.processed_out_tile_flat[((0*COLS + dbg_c)*DATA_WIDTH) +: DATA_WIDTH]));
            end

            $display("[NNIA TB][DEBUG] Row0 output-buffer tile values:");
            for (dbg_c = 0; dbg_c < COLS; dbg_c = dbg_c + 1) begin
                $display("  outbuf_row0[%0d]         = %0d",
                         dbg_c,
                         $signed(dut.outbuf_tile_flat[((0*COLS + dbg_c)*DATA_WIDTH) +: DATA_WIDTH]));
            end

            $display("[NNIA TB][DEBUG] Controller / tile state:");
            $display("  tile_k_base              = %0d", dut.tile_k_base);
            $display("  tile_n_base              = %0d", dut.tile_n_base);
            $display("  controller_state         = %0d", dut.u_tile_controller.state);
            $display("  stream_count             = %0d", dut.u_tile_controller.stream_count);
            $display("  drain_count              = %0d", dut.u_tile_controller.drain_count);
            $display("================================================================");
        end
    end

    // ------------------------------------------------------------------------
    // Clear local memories
    // ------------------------------------------------------------------------
    task clear_local_memories;
        integer idx;
        begin
            for (idx = 0; idx < INPUT_ELEMS; idx = idx + 1)
                input_mem[idx] = {DATA_WIDTH{1'b0}};

            for (idx = 0; idx < WEIGHT_ELEMS; idx = idx + 1)
                weight_mem[idx] = {DATA_WIDTH{1'b0}};

            for (idx = 0; idx < BIAS_ELEMS; idx = idx + 1)
                bias_mem[idx] = {DATA_WIDTH{1'b0}};

            for (idx = 0; idx < OUTPUT_ELEMS; idx = idx + 1) begin
                expected_mem[idx] = {DATA_WIDTH{1'b0}};
                rtl_out_mem[idx]  = {DATA_WIDTH{1'b0}};
            end
        end
    endtask

    // ------------------------------------------------------------------------
    // Load memory files
    // ------------------------------------------------------------------------
    task load_mem_files;
        begin
            fd = $fopen({MEM_BASE_PATH, "input.mem"}, "r");
            if (fd == 0) begin
                $display("[NNIA TB][FATAL] Unable to open %s", {MEM_BASE_PATH, "input.mem"});
                $finish;
            end
            $fclose(fd);

            fd = $fopen({MEM_BASE_PATH, "weights.mem"}, "r");
            if (fd == 0) begin
                $display("[NNIA TB][FATAL] Unable to open %s", {MEM_BASE_PATH, "weights.mem"});
                $finish;
            end
            $fclose(fd);

            fd = $fopen({MEM_BASE_PATH, "bias.mem"}, "r");
            if (fd == 0) begin
                $display("[NNIA TB][FATAL] Unable to open %s", {MEM_BASE_PATH, "bias.mem"});
                $finish;
            end
            $fclose(fd);

            fd = $fopen({MEM_BASE_PATH, "expected_output.mem"}, "r");
            if (fd == 0) begin
                $display("[NNIA TB][FATAL] Unable to open %s", {MEM_BASE_PATH, "expected_output.mem"});
                $finish;
            end
            $fclose(fd);

            $readmemh({MEM_BASE_PATH, "input.mem"},           input_mem);
            $readmemh({MEM_BASE_PATH, "weights.mem"},         weight_mem);
            $readmemh({MEM_BASE_PATH, "bias.mem"},            bias_mem);
            $readmemh({MEM_BASE_PATH, "expected_output.mem"}, expected_mem);

            $display("[NNIA TB][INFO] Reference memory files loaded successfully.");
            $display("[NNIA TB][INFO]   input.mem");
            $display("[NNIA TB][INFO]   weights.mem");
            $display("[NNIA TB][INFO]   bias.mem");
            $display("[NNIA TB][INFO]   expected_output.mem");
        end
    endtask

    // ------------------------------------------------------------------------
    // Utility helpers
    // ------------------------------------------------------------------------
    task wait_clocks;
        input integer num_cycles;
        integer k;
        begin
            for (k = 0; k < num_cycles; k = k + 1)
                @(posedge clk);
        end
    endtask

    task pulse_start;
        begin
            @(posedge clk);
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
        end
    endtask

    // ------------------------------------------------------------------------
    // Pack flattened DUT inputs
    // ------------------------------------------------------------------------
    task pack_flat_inputs;
        integer idx;
        begin
            input_flat  = {(M_TOTAL*K_TOTAL*DATA_WIDTH){1'b0}};
            weight_flat = {(K_TOTAL*N_TOTAL*DATA_WIDTH){1'b0}};
            bias_flat   = {(N_TOTAL*DATA_WIDTH){1'b0}};

            for (idx = 0; idx < INPUT_ELEMS; idx = idx + 1)
                input_flat[(idx*DATA_WIDTH) +: DATA_WIDTH] = input_mem[idx];

            for (idx = 0; idx < WEIGHT_ELEMS; idx = idx + 1)
                weight_flat[(idx*DATA_WIDTH) +: DATA_WIDTH] = weight_mem[idx];

            for (idx = 0; idx < BIAS_ELEMS; idx = idx + 1)
                bias_flat[(idx*DATA_WIDTH) +: DATA_WIDTH] = bias_mem[idx];
        end
    endtask

    // ------------------------------------------------------------------------
    // Wait for completion
    // ------------------------------------------------------------------------
    task wait_for_done_or_timeout;
        integer wait_cycles;
        begin
            wait_cycles = 0;

            while ((done !== 1'b1) &&
                   (error !== 1'b1) &&
                   (wait_cycles < MAX_CYCLES)) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
            end

            if (error === 1'b1) begin
                $display("[NNIA TB][FATAL] DUT asserted error before completion.");
                $finish;
            end
            else if (done !== 1'b1) begin
                $display("[NNIA TB][FATAL] Completion timeout after %0d cycles.", MAX_CYCLES);
                $finish;
            end
            else begin
                $display("[NNIA TB][INFO] Computation completed successfully after %0d wait cycles (global cycle=%0d).",
                         wait_cycles, cycle_count);
            end
        end
    endtask

    // ------------------------------------------------------------------------
    // Wait for output_valid after done
    // ------------------------------------------------------------------------
    task wait_for_output_valid_or_timeout;
        integer wait_cycles;
        begin
            wait_cycles = 0;

            while ((output_valid !== 1'b1) &&
                   (error !== 1'b1) &&
                   (wait_cycles < MAX_OUTPUT_WAIT)) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
            end

            if (error === 1'b1) begin
                $display("[NNIA TB][FATAL] DUT asserted error while waiting for output_valid.");
                $finish;
            end
            else if (output_valid !== 1'b1) begin
                $display("[NNIA TB][WARN] output_valid did not assert within %0d cycles after completion.",
                         MAX_OUTPUT_WAIT);
            end
            else begin
                $display("[NNIA TB][INFO] Output-valid asserted after %0d additional wait cycles.", wait_cycles);
            end
        end
    endtask

    // ------------------------------------------------------------------------
    // Capture RTL output
    // ------------------------------------------------------------------------
    task capture_rtl_outputs;
        integer idx;
        begin
            for (idx = 0; idx < OUTPUT_ELEMS; idx = idx + 1)
                rtl_out_mem[idx] = final_output_flat[(idx*DATA_WIDTH) +: DATA_WIDTH];
        end
    endtask

    // ------------------------------------------------------------------------
    // Write rtl_output.mem in fixed-width 16-bit HEX
    // ------------------------------------------------------------------------
    task write_rtl_output_file;
        integer idx;
        begin
            fd = $fopen({MEM_BASE_PATH, "rtl_output.mem"}, "w");
            if (fd == 0) begin
                $display("[NNIA TB][FATAL] Unable to open %s for write", {MEM_BASE_PATH, "rtl_output.mem"});
                $finish;
            end

            for (idx = 0; idx < OUTPUT_ELEMS; idx = idx + 1) begin
                raw_bits = rtl_out_mem[idx][DATA_WIDTH-1:0];
                $fdisplay(fd, "%04h", raw_bits);
            end

            $fclose(fd);
            $display("[NNIA TB][INFO] RTL output memory file generated successfully: %s",
                     {MEM_BASE_PATH, "rtl_output.mem"});
        end
    endtask

    // ------------------------------------------------------------------------
    // Print final output row0
    // ------------------------------------------------------------------------
    task print_final_row_debug;
        integer cidx;
        begin
            $display("----------------------------------------------------------------");
            $display("[NNIA TB][INFO] Final output snapshot | row 0");
            for (cidx = 0; cidx < N_TOTAL; cidx = cidx + 1) begin
                $display("  final_output[0,%0d] = %0d",
                         cidx,
                         $signed(final_output_flat[((0*N_TOTAL + cidx)*DATA_WIDTH) +: DATA_WIDTH]));
            end
            $display("----------------------------------------------------------------");
        end
    endtask

    // ------------------------------------------------------------------------
    // Print performance counters
    // These lines are intentionally KEY=VALUE format for easy parsing from
    // Vivado batch logs in VS Code / Python host flow.
    // ------------------------------------------------------------------------
    task print_perf_counters;
        begin
            $display("----------------------------------------------------------------");
            $display("[NNIA TB][INFO] Performance counter summary");
            $display("  perf_total_cycles = %0d", perf_total_cycles);
            $display("  perf_run_cycles   = %0d", perf_run_cycles);
            $display("  perf_run_count    = %0d", perf_run_count);
            $display("  perf_tile_writes  = %0d", perf_tile_writes);
            $display("  perf_psum_saves   = %0d", perf_psum_saves);
            $display("----------------------------------------------------------------");

            $display("PERF_TOTAL_CYCLES=%0d", perf_total_cycles);
            $display("PERF_RUN_CYCLES=%0d",   perf_run_cycles);
            $display("PERF_RUN_COUNT=%0d",    perf_run_count);
            $display("PERF_TILE_WRITES=%0d",  perf_tile_writes);
            $display("PERF_PSUM_SAVES=%0d",   perf_psum_saves);
        end
    endtask

    // ------------------------------------------------------------------------
    // Compare outputs
    // ------------------------------------------------------------------------
    task compare_outputs;
        integer idx;
        begin
            mismatch_count = 0;
            match_count    = 0;

            $display("----------------------------------------------------------------");
            $display("[NNIA TB][INFO] Starting RTL vs reference output comparison");
            $display("----------------------------------------------------------------");

            for (idx = 0; idx < OUTPUT_ELEMS; idx = idx + 1) begin
                actual_val   = rtl_out_mem[idx];
                expected_val = expected_mem[idx];

                if ((^actual_val === 1'bx) || (^expected_val === 1'bx)) begin
                    mismatch_count = mismatch_count + 1;
                    $display("[NNIA TB][INVALID] output[%0d] | expected=%0h | rtl=%0h",
                             idx, expected_val, actual_val);
                end
                else if (actual_val === expected_val) begin
                    match_count = match_count + 1;
                    $display("[NNIA TB][MATCH]   output[%0d] | expected=%0d | rtl=%0d",
                             idx, expected_val, actual_val);
                end
                else begin
                    mismatch_count = mismatch_count + 1;
                    $display("[NNIA TB][MISMATCH] output[%0d] | expected=%0d | rtl=%0d",
                             idx, expected_val, actual_val);
                end
            end

            $display("----------------------------------------------------------------");
            $display("[NNIA TB][INFO] Comparison summary");
            $display("  Total outputs checked : %0d", OUTPUT_ELEMS);
            $display("  Matching outputs      : %0d", match_count);
            $display("  Mismatched outputs    : %0d", mismatch_count);

            if (mismatch_count == 0)
                $display("[NNIA TB][PASS] Verification completed successfully. All outputs match the reference model.");
            else
                $display("[NNIA TB][FAIL] Verification completed with output mismatches.");

            $display("----------------------------------------------------------------");
        end
    endtask

    // ------------------------------------------------------------------------
    // Main stimulus
    // ------------------------------------------------------------------------
    initial begin
        rst         = 1'b1;
        start       = 1'b0;
        input_flat  = {(M_TOTAL*K_TOTAL*DATA_WIDTH){1'b0}};
        weight_flat = {(K_TOTAL*N_TOTAL*DATA_WIDTH){1'b0}};
        bias_flat   = {(N_TOTAL*DATA_WIDTH){1'b0}};

        clear_local_memories();

        $display("================================================================");
        $display("[NNIA TB] Neural Network Inference Accelerator | Full-System Verification");
        $display("================================================================");

        load_mem_files();
        pack_flat_inputs();

        $display("[NNIA TB][INFO] Applying reset sequence...");
        wait_clocks(5);
        rst = 1'b0;
        wait_clocks(2);

        $display("[NNIA TB][INFO] Launching inference run...");
        pulse_start();

        wait_clocks(1);
        if (busy !== 1'b1)
            $display("[NNIA TB][WARN] busy did not assert immediately after start pulse. Continuing with verification.");

        wait_for_done_or_timeout();
        wait_for_output_valid_or_timeout();

        // one extra cycle for stable final registered capture
        wait_clocks(1);

        if (output_valid)
            $display("[NNIA TB][INFO] Final output is marked valid at capture point.");
        else
            $display("[NNIA TB][WARN] Final output-valid flag is low at capture point. Proceeding with capture.");

        if (error_seen)
            $display("[NNIA TB][WARN] DUT asserted an error at some point during execution.");
        else
            $display("[NNIA TB][INFO] DUT completed without runtime error assertion.");

        print_final_row_debug();
        print_perf_counters();

        capture_rtl_outputs();
        write_rtl_output_file();
        compare_outputs();

        if (mismatch_count == 0)
            $display("[NNIA TB][FINAL PASS] End-to-end tiled NNIA inference matches the Python reference output.");
        else
            $display("[NNIA TB][FINAL FAIL] End-to-end tiled NNIA inference does not match the Python reference output.");

        $display("[NNIA TB][INFO] Simulation finished.");
        $finish;
    end

    // ------------------------------------------------------------------------
    // Global watchdog
    // ------------------------------------------------------------------------
    initial begin
        wd_count = 0;

        wait(!rst);

        while ((done !== 1'b1) && (error !== 1'b1) && (wd_count < MAX_CYCLES)) begin
            @(posedge clk);
            wd_count = wd_count + 1;
        end

        if ((done !== 1'b1) && (error !== 1'b1)) begin
            $display("[NNIA TB][FATAL] Global watchdog timeout.");
            $display("[NNIA TB][FATAL] Simulation exceeded %0d cycles without completion.", MAX_CYCLES);
            $finish;
        end
    end

endmodule
