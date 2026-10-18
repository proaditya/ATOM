# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
# FlyDSL kernel source, kept as written: it is traced at build time, where closures over loop
# variables and the nesting of dynamic conditions are deliberate (they shape the generated control flow).
"""Qwen3.8-27B verify megakernel, 8 token rows, long-context attention (FlyDSL, gfx950).

As ``verify_m8.py``, with the partial-attention stage split by row octets above 16K positions.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl, math as fmath, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir.dialects import llvm
from flydsl._mlir import ir

LDS_BYTES = 159744
NATIVE_WORDS = 3728
NATIVE_BYTES = 14912
CURRENT_WORD = 304
PEER_WORD = 400
CENSUS_WORD = 3472
CHUNK_WORD = 3728
MAX_COMMAND_EPOCH = 1048575

def build_verify_m8_long(layer_count=64, diagnostics=False):
    if type(layer_count) is not int or layer_count != 64:
        raise ValueError('literal full64 only')
    if diagnostics is not False:
        raise ValueError('357 fixed primary final-only')
    kernel_name = 'qwen38_verify_m8_long'

    @fx.struct
    class Smem:
        arena: fx.Array[fx.Uint8, LDS_BYTES, 16]
        status: fx.Array[fx.Int32, 12, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[256, 1, 1])
    def kernel(pointer_table: fx.Pointer):
        tid = fx.thread_idx.x
        block = fx.block_idx.x
        allocator = fx.SharedAllocator()
        storage = allocator.allocate(Smem).peek()
        arena = storage.arena.ptr
        status_ptr = storage.status.ptr
        alive = fx.Int32(1)

        def raw_table_address(row, slot):
            return fx.Uint64(llvm.inline_asm(T.i64, [fx.Uint64(fx.ptrtoint(pointer_table + row * 45 + slot)).ir_value()], 's_load_dwordx2 $0, $1, 0x0\ns_waitcnt lgkmcnt(0)', '=&s,s,~{memory}', has_side_effects=True))

        def endpoint_address(slot):
            return raw_table_address(fx.Int32(64), slot)

        native = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 64), endpoint_address(42))
        native_u64 = fx.inttoptr(fx.PointerType.get(fx.Uint64.ir_type, fx.AddressSpace.Global, 64), fx.ptrtoint(native))

        def coherent_u32(p):
            return fx.Int32(llvm.inline_asm(T.i32, [fx.Uint64(fx.ptrtoint(p)).ir_value()], 'global_load_dword $0, $1, off sc0 sc1\ns_waitcnt vmcnt(0)', '=&v,v,~{memory}', has_side_effects=True))

        def coherent_u64(p):
            raw = fx.Uint64(llvm.inline_asm(T.i64, [fx.Uint64(fx.ptrtoint(p)).ir_value()], 'global_load_dwordx2 $0, $1, off sc0 sc1\ns_waitcnt vmcnt(0)', '=&v,v,~{memory}', has_side_effects=True))
            lo = fx.Uint32(rocdl.readfirstlane(T.i32, raw.to(fx.Uint32).ir_value()))
            hi = fx.Uint32(rocdl.readfirstlane(T.i32, raw.shrui(fx.Uint64(32)).to(fx.Uint32).ir_value()))
            return lo.to(fx.Uint64) | (hi.to(fx.Uint64) << fx.Uint64(32))

        def system_store_u32(p, value):
            llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(p)).ir_value(), fx.Int32(value).ir_value()], 'global_store_dword $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)

        def system_store_u64(p, value):
            llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(p)).ir_value(), fx.Uint64(value).ir_value()], 'global_store_dwordx2 $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)

        def system_atomic_add(p):
            return fx.Int32(llvm.inline_asm(T.i32, [fx.Uint64(fx.ptrtoint(p)).ir_value(), fx.Int32(1).ir_value()], 'flat_atomic_add $0, $1, $2 sc0 sc1\ns_waitcnt vmcnt(0) lgkmcnt(0)', '=&v,v,v,~{memory}', has_side_effects=True))

        def system_atomic_add_nw(p):
            # system-scope increment, no return value, no wait (publisher drained its data before).
            llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(p)).ir_value(), fx.Int32(1).ir_value()], 'global_atomic_add $0, $1, off sc1', 'v,v,~{memory}', has_side_effects=True)

        def fail_native():
            prior, won = fx.atomic_cas(native + 289, fx.Int32(0), fx.Int32(2), syncscope='agent', success_ordering=fx.AtomicOrdering.AcqRel, failure_ordering=fx.AtomicOrdering.Acquire)

        xcc = fx.Int32(llvm.inline_asm(T.i32, [], 's_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 16)', '=s,~{memory}', has_side_effects=True))
        command_epoch = coherent_u32(native + 288)
        if tid == 0:
            system_store_u32(native + CENSUS_WORD + block, xcc)
            if (xcc < 0) | (xcc >= 8) | (command_epoch < 1) | (command_epoch > MAX_COMMAND_EPOCH):
                fail_native()
        # Bad topology/epochs cannot index release storage or overflow generation.
        safe_xcc = fx.max(fx.Int32(0), fx.min(xcc, fx.Int32(7)))
        safe_command = fx.max(fx.Int32(1), fx.min(command_epoch, fx.Int32(MAX_COMMAND_EPOCH)))
        command_base = safe_command * fx.Int32(2048)

        def release_epochs(epoch):
            for release_xcc in range_constexpr(8):
                system_store_u32(native + 144 + release_xcc * 16, epoch)
            llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)

        def native_wait(epoch, status_slot, consume):
            if tid == 0:
                ready = fx.Int32(0)
                failed = coherent_u32(native + 289)
                polls = fx.Int32(0)
                while (ready == 0) & (failed == 0) & (polls < 1048576):
                    ready = (coherent_u32(native + 144 + safe_xcc * 16) >= epoch).to(fx.Int32)
                    failed = coherent_u32(native + 289)
                    llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                    polls = polls + 1
                answer = fx.Int32(1)
                if (ready == 0) | (failed != 0):
                    fail_native()
                    answer = fx.Int32(2)
                if (answer == 1) & consume:
                    llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)\nbuffer_inv sc1', '~{memory}', has_side_effects=True)
                fx.ptr_store(answer, status_ptr + status_slot)
            gpu.barrier()
            return fx.Int32(fx.ptr_load(status_ptr + status_slot))

        def native_arrive(epoch):
            # Every wave and every CTA participates, including inactive jobs.
            llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
            gpu.barrier()
            if tid == 0:
                local_prior = fx.atomic_add(native + safe_xcc * 16, fx.Int32(1), ordering=fx.AtomicOrdering.Monotonic, syncscope='agent')
                if (local_prior + 1) % 32 == 0:
                    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope='agent')
                    global_prior = system_atomic_add(native + 128)
                    if (global_prior + 1) % 8 == 0:
                        release_epochs(epoch)

        def native_grid(epoch, status_slot, consumer_blocks=256):
            native_arrive(epoch)
            return native_wait(epoch, status_slot, block < consumer_blocks)

        def native_postnorm(epoch, status_slot, layer_index):
            norm_owner = (block == 0) | ((block >= 137) & (block <= 143))
            pn_result = fx.Int32(2)
            if norm_owner:
                llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                gpu.barrier()
                if tid == 0:
                    system_atomic_add_nw(native + 272)
                    # wait for all owners, then publish to this XCD's local release word.
                    ol_target = (layer_index + 1) * 8
                    ol_ready = fx.Int32(0)
                    ol_failed = coherent_u32(native + 289)
                    ol_polls = fx.Int32(0)
                    while (ol_ready == 0) & (ol_failed == 0) & (ol_polls < 1048576):
                        ol_ready = (coherent_u32(native + 272) >= ol_target).to(fx.Int32)
                        ol_failed = coherent_u32(native + 289)
                        llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                        ol_polls = ol_polls + 1
                    ol_answer = fx.Int32(1)
                    if (ol_ready == 0) | (ol_failed != 0):
                        fail_native()
                        ol_answer = fx.Int32(2)
                    llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)\nbuffer_inv sc1', '~{memory}', has_side_effects=True)
                    fx.ptr_store(layer_index + 1, native + CHUNK_WORD + 832 + safe_xcc * 16)
                    llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                    fx.ptr_store(ol_answer, status_ptr + status_slot)
                gpu.barrier()
                pn_result = fx.Int32(fx.ptr_load(status_ptr + status_slot))
            else:
                pn_result = native_wait(epoch, status_slot, (block > 0) & ((block <= 136) | (block >= 144)))
            return pn_result

        def scalar_word(p):
            sw_addr = fx.Uint64(fx.ptrtoint(p))
            sw_lo = fx.Uint32(rocdl.readfirstlane(T.i32, sw_addr.to(fx.Uint32).ir_value()))
            sw_hi = fx.Uint32(rocdl.readfirstlane(T.i32, sw_addr.shrui(fx.Uint64(32)).to(fx.Uint32).ir_value()))
            sw_saddr = sw_lo.to(fx.Uint64) | (sw_hi.to(fx.Uint64) << fx.Uint64(32))
            return fx.Int32(llvm.inline_asm(T.i32, [sw_saddr.ir_value()], 's_dcache_inv\ns_load_dword $0, $1, 0x0 glc\ns_waitcnt lgkmcnt(0)', '=s,s,~{memory}', has_side_effects=True))

        def up_local_release(status_slot, layer_index):
            # Up CTAs: scalar poll of the XCD-local release word (no vmcnt coupling).
            if tid == 0:
                ul_target = layer_index + 1
                ul_ready = fx.Int32(0)
                ul_polls = fx.Int32(0)
                while (ul_ready == 0) & (ul_polls < 4194304):
                    ul_addr = fx.Uint64(fx.ptrtoint(native + CHUNK_WORD + 832 + safe_xcc * 16))
                    ul_lo = fx.Uint32(rocdl.readfirstlane(T.i32, ul_addr.to(fx.Uint32).ir_value()))
                    ul_hi = fx.Uint32(rocdl.readfirstlane(T.i32, ul_addr.shrui(fx.Uint64(32)).to(fx.Uint32).ir_value()))
                    ul_saddr = ul_lo.to(fx.Uint64) | (ul_hi.to(fx.Uint64) << fx.Uint64(32))
                    ul_v = fx.Int32(llvm.inline_asm(T.i32, [ul_saddr.ir_value()], 's_dcache_inv\ns_load_dword $0, $1, 0x0 glc\ns_waitcnt lgkmcnt(0)', '=s,s,~{memory}', has_side_effects=True))
                    ul_ready = (ul_v >= ul_target).to(fx.Int32)
                    llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                    ul_polls = ul_polls + 1
                ul_answer = fx.Int32(1)
                if ul_ready == 0:
                    fail_native()
                    ul_answer = fx.Int32(2)
                fx.ptr_store(ul_answer, status_ptr + status_slot)
            gpu.barrier()
            return fx.Int32(fx.ptr_load(status_ptr + status_slot))

        def rms_payload(hidden, residual, gamma):
            raw = llvm.inline_asm(ir.Type.parse('!llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>)>'), [fx.Uint64(fx.ptrtoint(hidden)).ir_value(), fx.Uint64(fx.ptrtoint(gamma)).ir_value(), fx.Uint64(fx.ptrtoint(residual)).ir_value()], 'global_load_dwordx4 $0, $3, off sc0\nglobal_load_dwordx4 $1, $4, off\nglobal_load_dwordx4 $2, $5, off sc0\ns_waitcnt vmcnt(0)', '=&v,=&v,=&v,v,v,v,~{memory}', has_side_effects=True)
            return (fx.Vector(llvm.extractvalue(T.vec(4, T.i32), raw, [0])).bitcast(fx.BFloat16), fx.Vector(llvm.extractvalue(T.vec(4, T.i32), raw, [1])).bitcast(fx.BFloat16), fx.Vector(llvm.extractvalue(T.vec(4, T.i32), raw, [2])).bitcast(fx.BFloat16))

        def fold_payload(partial, residual, gamma):
            raw = llvm.inline_asm(ir.Type.parse('!llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>'), [fx.Uint64(fx.ptrtoint(partial)).ir_value(), fx.Uint64(fx.ptrtoint(partial + 40960)).ir_value(), fx.Uint64(fx.ptrtoint(partial + 81920)).ir_value(), fx.Uint64(fx.ptrtoint(gamma)).ir_value(), fx.Uint64(fx.ptrtoint(residual)).ir_value()], 'global_load_dwordx4 $0, $8, off sc0\nglobal_load_dwordx4 $1, $8, off offset:16 sc0\nglobal_load_dwordx4 $2, $9, off sc0\nglobal_load_dwordx4 $3, $9, off offset:16 sc0\nglobal_load_dwordx4 $4, $10, off sc0\nglobal_load_dwordx4 $5, $10, off offset:16 sc0\nglobal_load_dwordx4 $6, $11, off\nglobal_load_dwordx4 $7, $12, off sc0\ns_waitcnt vmcnt(0)', '=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,v,v,v,v,v,~{memory}', has_side_effects=True)
            vectors = [fx.Vector(llvm.extractvalue(T.vec(4, T.i32), raw, [i])) for i in range_constexpr(8)]
            planes = []
            for p in range_constexpr(3):
                planes.append(fx.Vector.from_elements([vectors[p*2][i] for i in range_constexpr(4)] + [vectors[p*2+1][i] for i in range_constexpr(4)], fx.Int32).bitcast(fx.Float32))
            return planes[0], planes[1], planes[2], vectors[6].bitcast(fx.BFloat16), vectors[7].bitcast(fx.BFloat16)

        def rms_store(p, value):
            packed = value.bitcast(fx.Int32)
            llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(p)).ir_value(), packed.ir_value()], 'global_store_dwordx4 $0, $1, off sc0 nt\ns_waitcnt vmcnt(0)', 'v,v,~{memory}', has_side_effects=True)

        def rms_pair_store(residual_pointer, residual_value, normalized_pointer, normalized_value):
            packed_residual = residual_value.bitcast(fx.Int32)
            packed_normalized = normalized_value.bitcast(fx.Int32)
            # Both addresses/data stay operands of this one asm through its final VM0.
            llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(residual_pointer)).ir_value(), packed_residual.ir_value(), fx.Uint64(fx.ptrtoint(normalized_pointer)).ir_value(), packed_normalized.ir_value()], 'global_store_dwordx4 $0, $1, off sc0 nt\nglobal_store_dwordx4 $2, $3, off sc0 nt\ns_waitcnt vmcnt(0)', 'v,v,v,v,~{memory}', has_side_effects=True)

        def nrm_desc(p, size):
            nrm_addr = fx.Int64(fx.ptrtoint(p))
            return fx.Vector.from_elements([fx.Int32(rocdl.readfirstlane(T.i32, nrm_addr.to(fx.Int32).ir_value())), fx.Int32(rocdl.readfirstlane(T.i32, nrm_addr.shrui(fx.Int64(32)).to(fx.Int32).ir_value())), fx.Int32(size), fx.Int32(159744)], fx.Int32)

        def nrm_ld16(desc, offset, aux):
            return fx.Vector(llvm.call_intrinsic(T.vec(4, T.i32), 'llvm.amdgcn.raw.buffer.load.v4i32', [desc.ir_value(), fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(aux).ir_value()], [], []))

        def nrm_st16(desc, offset, value, aux):
            llvm.call_intrinsic(None, 'llvm.amdgcn.raw.buffer.store.v4i32', [value.ir_value(), desc.ir_value(), fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(aux).ir_value()], [], [])

        def nrm_offset(valid, byte_offset):
            # Invalid columns move past num_records: loads return 0, stores are dropped.
            return fx.Int32(byte_offset) + (fx.Int32(1) - valid.to(fx.Int32)) * fx.Int32(0x10000000)

        def native_entry_census():
            # One-use topology admission precedes every ordinary XCC election.
            llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
            gpu.barrier()
            if tid == 0:
                # Completed system atomics order the census store before arrival.
                xcc_prior = system_atomic_add(native + 293 + safe_xcc)
                census_prior = system_atomic_add(native + 291)
                if census_prior + 1 == 256:
                    topology_ok = fx.Int32(1)
                    for census_xcc in range_constexpr(8):
                        if coherent_u32(native + 293 + census_xcc) != 32:
                            topology_ok = fx.Int32(2)
                    if coherent_u32(native + 289) != 0:
                        topology_ok = fx.Int32(2)
                    if topology_ok != 1:
                        fail_native()
                    system_store_u32(native + 292, topology_ok)
                    llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                gate_answer = fx.Int32(0)
                gate_failed = coherent_u32(native + 289)
                gate_polls = fx.Int32(0)
                while (gate_answer == 0) & (gate_failed == 0) & (gate_polls < 1048576):
                    gate_answer = coherent_u32(native + 292)
                    gate_failed = coherent_u32(native + 289)
                    llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                    gate_polls = gate_polls + 1
                if (gate_answer != 1) | (gate_failed != 0):
                    fail_native()
                    gate_answer = fx.Int32(2)
                fx.ptr_store(gate_answer, status_ptr + 11)
            gpu.barrier()
            return fx.Int32(fx.ptr_load(status_ptr + 11))

        alive = fx.max(alive, native_entry_census())

        # Common endpoints are explicit complete-token work, not native decoder-only.
        ids = fx.inttoptr(fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 16), endpoint_address(44))
        embedding = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), endpoint_address(43))
        hidden0 = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), raw_table_address(fx.Int32(0), 0))
        residual0 = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), raw_table_address(fx.Int32(0), 1))
        initial_mixer = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), raw_table_address(fx.Int32(0), 32))
        if (alive == 1) & (block < 8):
            token = fx.ptr_load(ids + block)
            for step in range_constexpr(3):
                column = tid * 8 + step * 2048
                if column < 5120:
                    value = fx.Vector.filled(8, 0.0, fx.BFloat16)
                    if (token >= 0) & (token < 248320):
                        value = fx.Vector(fx.ptr_load(embedding + token * 5120 + column, result_type=T.vec(8, T.bf16)))
                    fx.ptr_store(value, hidden0 + block * 5120 + column)
                    fx.ptr_store(fx.Vector.filled(8, 0.0, fx.BFloat16), residual0 + block * 5120 + column)
        initial_index = block * 256 + tid
        if (alive == 1) & (initial_index < 40960):
            fx.ptr_store(fx.BFloat16(0.0), initial_mixer + initial_index)
        if alive == 1:
            alive = fx.max(alive, native_grid(command_base + 1, 11))

        layer = fx.Int32(0)
        while (layer < 64) & (alive == 1):
            # Failure extension: retire prior LDS status readers before reuse.
            gpu.barrier()
            if tid == 0:
                entry_answer = fx.Int32(1)
                if (alive != 1) | (coherent_u32(native + 289) != 0):
                    entry_answer = fx.Int32(2)
                fx.ptr_store(entry_answer, status_ptr + 11)
            gpu.barrier()
            alive = fx.max(alive, fx.Int32(fx.ptr_load(status_ptr + 11)))
            iteration_tid = fx.Int32(llvm.inline_asm(T.i32, [tid.ir_value()], '', '=v,0', has_side_effects=True)) & fx.Int32(255)
            iteration_block = fx.Int32(llvm.inline_asm(T.i32, [block.ir_value()], '', '=s,0', has_side_effects=True)) & fx.Int32(255)
            epoch_base = command_base + 2 + layer * 12 - layer // 4
            if (block == 0) & (tid == 0) & (alive == 1) & (layer == 63):
                # Prior successful grid retired all old binding readers; on abort
                # publisher must not advance the shared mutable record.
                if coherent_u32(native + 289) == 0:
                    for current_slot in range_constexpr(45):
                        system_store_u64(native_u64 + CURRENT_WORD // 2 + current_slot, raw_table_address(layer, current_slot))

            def table_address(slot):
                return raw_table_address(layer, slot)

            control = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16), raw_table_address(layer, 42))

            def rms_stage(rms_hidden, rms_residual, rms_shifted_gamma, rms_normalized, rms_residual_out, rms_activation, rms_scales, rms_owner):
                (rms_row, rms_tid) = (rms_owner, iteration_tid)
                (rms_lane, rms_wave) = (rms_tid % 64, rms_tid // 64)
                rms_partials = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                llvm.InlineAsmOp(None, [], 's_setreg_imm32_b32 0xdc1, 1;', '~{memory}', has_side_effects=True)
                rms_saved_data = [fx.Vector.filled(8, 0.0, fx.BFloat16) for _ in range_constexpr(3)]
                rms_saved_skip = [fx.Vector.filled(8, 0.0, fx.BFloat16) for _ in range_constexpr(3)]
                rms_saved_gamma = [fx.Vector.filled(8, 0.0, fx.BFloat16) for _ in range_constexpr(3)]
                rms_acc = fx.Vector.filled(8, 0.0, fx.Float32)
                rms_hd = nrm_desc(rms_hidden, 81920)
                rms_rd = nrm_desc(rms_residual, 81920)
                rms_gd = nrm_desc(rms_shifted_gamma, 10240)
                rms_od = nrm_desc(rms_residual_out, 81920)
                rms_nd = nrm_desc(rms_normalized, 81920)
                rms_loaded = []
                for rms_step in range_constexpr(3):
                    rms_column = rms_step * 2048 + rms_tid * 8
                    rms_valid = rms_column < 5120
                    rms_loaded.append((nrm_ld16(rms_hd, nrm_offset(rms_valid, (rms_row * 5120 + rms_column) * 2), 1), nrm_ld16(rms_gd, nrm_offset(rms_valid, rms_column * 2), 0), nrm_ld16(rms_rd, nrm_offset(rms_valid, (rms_row * 5120 + rms_column) * 2), 1)))
                for rms_step in range_constexpr(3):
                    rms_data = rms_loaded[rms_step][0].bitcast(fx.BFloat16)
                    loaded_gamma = rms_loaded[rms_step][1].bitcast(fx.BFloat16)
                    rms_skip = rms_loaded[rms_step][2].bitcast(fx.BFloat16)

                    rms_saved_data[rms_step] = rms_data
                    rms_saved_skip[rms_step] = rms_skip
                    rms_saved_gamma[rms_step] = loaded_gamma
                    rms_summed = rms_data.to(fx.Float32) + rms_skip.to(fx.Float32)
                    rms_acc = fx.Vector(fmath.fma(rms_summed, rms_summed, rms_acc))
                rms_total = rms_acc[0] + rms_acc[1] + (rms_acc[2] + rms_acc[3]) + (rms_acc[4] + rms_acc[5] + (rms_acc[6] + rms_acc[7]))
                rms_total = fx.Float32(llvm.inline_asm(T.f32, [rms_total.ir_value()], 's_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n', '=v,0', has_side_effects=True))
                rms_total = rms_total + rms_total.shuffle_xor(16, 64)
                rms_total = rms_total + rms_total.shuffle_xor(32, 64)
                if rms_lane == 0:
                    fx.ptr_store(rms_total, rms_partials + rms_wave)
                gpu.barrier()
                rms_total = fx.Float32(0.0)
                if rms_lane < 4:
                    rms_total = fx.Float32(fx.ptr_load(rms_partials + rms_lane))
                rms_total = fx.Float32(llvm.inline_asm(T.f32, [rms_total.ir_value()], 's_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n', '=v,0', has_side_effects=True))
                rms_total = fx.Int32(rocdl.readfirstlane(T.i32, rms_total.bitcast(fx.Int32).ir_value())).bitcast(fx.Float32)
                rms_variance_eps = fx.Float32(fmath.fma(rms_total, fx.Float32(1.0 / 5120.0), fx.Float32(1e-06)))
                rms_rms = fx.Float32(rocdl.rsq(T.f32, rms_variance_eps.ir_value()))
                for rms_step in range_constexpr(3):
                    rms_column = rms_step * 2048 + rms_tid * 8
                    rms_data = fx.Vector.filled(8, 0.0, fx.BFloat16)
                    rms_skip = fx.Vector.filled(8, 0.0, fx.BFloat16)
                    rms_gamma = fx.Vector.filled(8, 0.0, fx.BFloat16)
                    if rms_column < 5120:
                        rms_data = rms_saved_data[rms_step]
                        rms_skip = rms_saved_skip[rms_step]
                        rms_gamma = rms_saved_gamma[rms_step]
                    rms_rounded_sum = (rms_data.to(fx.Float32) + rms_skip.to(fx.Float32)).to(fx.BFloat16)
                    rms_norm = (rms_rounded_sum.to(fx.Float32) * rms_rms * rms_gamma.to(fx.Float32)).to(fx.BFloat16)
                    rms_store_offset = nrm_offset(rms_column < 5120, (rms_row * 5120 + rms_column) * 2)
                    nrm_st16(rms_od, rms_store_offset, rms_rounded_sum.bitcast(fx.Int32), 17)
                    nrm_st16(rms_nd, rms_store_offset, rms_norm.bitcast(fx.Int32), 17)
                    rms_raw = rms_norm.bitcast(fx.Int32)
                    rms_values = rms_norm.to(fx.Float32)
                    rms_amax = fx.Float32(0.0)
                    for rms_element in range_constexpr(8):
                        rms_amax = fx.max(rms_amax, fmath.absf(rms_values[rms_element]))
                    for rms_shift in (1, 2):
                        rms_amax = fx.max(rms_amax, rms_amax.shuffle_xor(rms_shift, 64))
                    rms_bits = (rms_amax * fx.Float32(1.0 / 448.0)).bitcast(fx.Int32)
                    rms_exponent = rms_bits.shrui(fx.Int32(23)) & fx.Int32(255)
                    if rms_bits & fx.Int32(8388607) != 0:
                        rms_exponent = rms_exponent + 1
                    rms_exponent = fx.max(fx.Int32(1), fx.min(rms_exponent, fx.Int32(254)))
                    rms_hw_scale = (rms_exponent << fx.Int32(23)).bitcast(fx.Float32)
                    rms_qw0 = fx.Int32(llvm.inline_asm(T.i32, [rms_raw[0].ir_value(), rms_hw_scale.ir_value(), fx.Int32(0).ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2', '=v,v,v,0', has_side_effects=True))
                    rms_qw0 = fx.Int32(llvm.inline_asm(T.i32, [rms_raw[1].ir_value(), rms_hw_scale.ir_value(), rms_qw0.ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2 op_sel:[0,0,1]', '=v,v,v,0', has_side_effects=True))
                    rms_qw1 = fx.Int32(llvm.inline_asm(T.i32, [rms_raw[2].ir_value(), rms_hw_scale.ir_value(), fx.Int32(0).ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2', '=v,v,v,0', has_side_effects=True))
                    rms_qw1 = fx.Int32(llvm.inline_asm(T.i32, [rms_raw[3].ir_value(), rms_hw_scale.ir_value(), rms_qw1.ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2 op_sel:[0,0,1]', '=v,v,v,0', has_side_effects=True))
                    if rms_column < 5120:
                        rms_output = fx.Vector.from_elements([rms_qw0, rms_qw1], fx.Int32).bitcast(fx.Uint8)
                        llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(rms_activation + rms_row * 5120 + rms_column)).ir_value(), fx.Vector.from_elements([rms_qw0, rms_qw1], fx.Int32).ir_value()], 'global_store_dwordx2 $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                        if rms_tid % 4 == 0:
                            rms_qb = rms_column // 32
                            llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(rms_scales + rms_qb // 4 * 64 + rms_qb % 4 * 16 + rms_row)).ir_value(), fx.Int32(rms_exponent).ir_value()], 'global_store_byte $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                if rms_row == 0:
                    for rms_step in range_constexpr(11):
                        rms_index = rms_tid + rms_step * 256
                        if rms_index < 2752:
                            if rms_index >= 2560 or rms_index % 16 >= 8:
                                llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(rms_scales + rms_index)).ir_value(), fx.Int32(127).ir_value()], 'global_store_byte $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)

            def fold_rms_stage(foldrms_partial, foldrms_down, foldrms_residual, foldrms_shifted_gamma, foldrms_normalized, foldrms_residual_out, foldrms_activation, foldrms_scales):
                (foldrms_row, foldrms_tid) = (iteration_block, iteration_tid)
                (foldrms_lane, foldrms_wave) = (foldrms_tid % 64, foldrms_tid // 64)
                foldrms_partials = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                llvm.InlineAsmOp(None, [], 's_setreg_imm32_b32 0xdc1, 1;', '~{memory}', has_side_effects=True)
                foldrms_saved_data = [fx.Vector.filled(8, 0.0, fx.BFloat16) for _ in range_constexpr(3)]
                foldrms_saved_skip = [fx.Vector.filled(8, 0.0, fx.BFloat16) for _ in range_constexpr(3)]
                foldrms_saved_gamma = [fx.Vector.filled(8, 0.0, fx.BFloat16) for _ in range_constexpr(3)]
                foldrms_acc = fx.Vector.filled(8, 0.0, fx.Float32)
                foldrms_pd = nrm_desc(foldrms_partial, 491520)
                foldrms_rd = nrm_desc(foldrms_residual, 81920)
                foldrms_gd = nrm_desc(foldrms_shifted_gamma, 10240)
                foldrms_od = nrm_desc(foldrms_residual_out, 81920)
                foldrms_nd = nrm_desc(foldrms_normalized, 81920)
                foldrms_loaded = []
                for foldrms_step in range_constexpr(3):
                    foldrms_column = foldrms_step * 2048 + foldrms_tid * 8
                    foldrms_valid = foldrms_column < 5120
                    foldrms_offset = foldrms_row * 5120 + foldrms_column
                    foldrms_parts = [nrm_ld16(foldrms_pd, nrm_offset(foldrms_valid, (foldrms_plane * 40960 + foldrms_offset) * 4 + foldrms_half * 16), 1) for foldrms_plane in range_constexpr(3) for foldrms_half in range_constexpr(2)]
                    foldrms_loaded.append((foldrms_parts, nrm_ld16(foldrms_gd, nrm_offset(foldrms_valid, foldrms_column * 2), 0), nrm_ld16(foldrms_rd, nrm_offset(foldrms_valid, foldrms_offset * 2), 1)))
                for foldrms_step in range_constexpr(3):
                    foldrms_column = foldrms_step * 2048 + foldrms_tid * 8
                    foldrms_parts, foldrms_g, foldrms_s = foldrms_loaded[foldrms_step]
                    foldrms_planes = [fx.Vector.from_elements([foldrms_parts[p * 2][i] for i in range_constexpr(4)] + [foldrms_parts[p * 2 + 1][i] for i in range_constexpr(4)], fx.Int32).bitcast(fx.Float32) for p in range_constexpr(3)]
                    loaded_gamma = foldrms_g.bitcast(fx.BFloat16)
                    foldrms_skip = foldrms_s.bitcast(fx.BFloat16)
                    foldrms_data = (foldrms_planes[0] + foldrms_planes[1] + foldrms_planes[2]).to(fx.BFloat16)
                    if fx.const_expr(diagnostics):
                        if foldrms_column < 5120:
                            rms_store(foldrms_down + foldrms_row * 5120 + foldrms_column, foldrms_data)

                    foldrms_saved_data[foldrms_step] = foldrms_data
                    foldrms_saved_skip[foldrms_step] = foldrms_skip
                    foldrms_saved_gamma[foldrms_step] = loaded_gamma
                    foldrms_summed = foldrms_data.to(fx.Float32) + foldrms_skip.to(fx.Float32)
                    foldrms_acc = fx.Vector(fmath.fma(foldrms_summed, foldrms_summed, foldrms_acc))
                foldrms_total = foldrms_acc[0] + foldrms_acc[1] + (foldrms_acc[2] + foldrms_acc[3]) + (foldrms_acc[4] + foldrms_acc[5] + (foldrms_acc[6] + foldrms_acc[7]))
                foldrms_total = fx.Float32(llvm.inline_asm(T.f32, [foldrms_total.ir_value()], 's_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n', '=v,0', has_side_effects=True))
                foldrms_total = foldrms_total + foldrms_total.shuffle_xor(16, 64)
                foldrms_total = foldrms_total + foldrms_total.shuffle_xor(32, 64)
                if foldrms_lane == 0:
                    fx.ptr_store(foldrms_total, foldrms_partials + foldrms_wave)
                gpu.barrier()
                foldrms_total = fx.Float32(0.0)
                if foldrms_lane < 4:
                    foldrms_total = fx.Float32(fx.ptr_load(foldrms_partials + foldrms_lane))
                foldrms_total = fx.Float32(llvm.inline_asm(T.f32, [foldrms_total.ir_value()], 's_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n', '=v,0', has_side_effects=True))
                foldrms_total = fx.Int32(rocdl.readfirstlane(T.i32, foldrms_total.bitcast(fx.Int32).ir_value())).bitcast(fx.Float32)
                foldrms_variance_eps = fx.Float32(fmath.fma(foldrms_total, fx.Float32(1.0 / 5120.0), fx.Float32(1e-06)))
                foldrms_rms = fx.Float32(rocdl.rsq(T.f32, foldrms_variance_eps.ir_value()))
                for foldrms_step in range_constexpr(3):
                    foldrms_column = foldrms_step * 2048 + foldrms_tid * 8
                    foldrms_data = fx.Vector.filled(8, 0.0, fx.BFloat16)
                    foldrms_skip = fx.Vector.filled(8, 0.0, fx.BFloat16)
                    foldrms_gamma = fx.Vector.filled(8, 0.0, fx.BFloat16)
                    if foldrms_column < 5120:
                        foldrms_data = foldrms_saved_data[foldrms_step]
                        foldrms_skip = foldrms_saved_skip[foldrms_step]
                        foldrms_gamma = foldrms_saved_gamma[foldrms_step]
                    foldrms_rounded_sum = (foldrms_data.to(fx.Float32) + foldrms_skip.to(fx.Float32)).to(fx.BFloat16)
                    foldrms_norm = (foldrms_rounded_sum.to(fx.Float32) * foldrms_rms * foldrms_gamma.to(fx.Float32)).to(fx.BFloat16)
                    foldrms_store_offset = nrm_offset(foldrms_column < 5120, (foldrms_row * 5120 + foldrms_column) * 2)
                    nrm_st16(foldrms_od, foldrms_store_offset, foldrms_rounded_sum.bitcast(fx.Int32), 3)
                    nrm_st16(foldrms_nd, foldrms_store_offset, foldrms_norm.bitcast(fx.Int32), 3)
                    foldrms_raw = foldrms_norm.bitcast(fx.Int32)
                    foldrms_values = foldrms_norm.to(fx.Float32)
                    foldrms_amax = fx.Float32(0.0)
                    for foldrms_element in range_constexpr(8):
                        foldrms_amax = fx.max(foldrms_amax, fmath.absf(foldrms_values[foldrms_element]))
                    for foldrms_shift in (1, 2):
                        foldrms_amax = fx.max(foldrms_amax, foldrms_amax.shuffle_xor(foldrms_shift, 64))
                    foldrms_bits = (foldrms_amax * fx.Float32(1.0 / 448.0)).bitcast(fx.Int32)
                    foldrms_exponent = foldrms_bits.shrui(fx.Int32(23)) & fx.Int32(255)
                    if foldrms_bits & fx.Int32(8388607) != 0:
                        foldrms_exponent = foldrms_exponent + 1
                    foldrms_exponent = fx.max(fx.Int32(1), fx.min(foldrms_exponent, fx.Int32(254)))
                    foldrms_hw_scale = (foldrms_exponent << fx.Int32(23)).bitcast(fx.Float32)
                    foldrms_qw0 = fx.Int32(llvm.inline_asm(T.i32, [foldrms_raw[0].ir_value(), foldrms_hw_scale.ir_value(), fx.Int32(0).ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2', '=v,v,v,0', has_side_effects=True))
                    foldrms_qw0 = fx.Int32(llvm.inline_asm(T.i32, [foldrms_raw[1].ir_value(), foldrms_hw_scale.ir_value(), foldrms_qw0.ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2 op_sel:[0,0,1]', '=v,v,v,0', has_side_effects=True))
                    foldrms_qw1 = fx.Int32(llvm.inline_asm(T.i32, [foldrms_raw[2].ir_value(), foldrms_hw_scale.ir_value(), fx.Int32(0).ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2', '=v,v,v,0', has_side_effects=True))
                    foldrms_qw1 = fx.Int32(llvm.inline_asm(T.i32, [foldrms_raw[3].ir_value(), foldrms_hw_scale.ir_value(), foldrms_qw1.ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2 op_sel:[0,0,1]', '=v,v,v,0', has_side_effects=True))
                    if foldrms_column < 5120:
                        foldrms_output = fx.Vector.from_elements([foldrms_qw0, foldrms_qw1], fx.Int32).bitcast(fx.Uint8)
                        fx.ptr_store(foldrms_output, foldrms_activation + foldrms_row * 5120 + foldrms_column)
                        if foldrms_tid % 4 == 0:
                            foldrms_qb = foldrms_column // 32
                            fx.ptr_store(foldrms_exponent.to(fx.Uint8), foldrms_scales + foldrms_qb // 4 * 64 + foldrms_qb % 4 * 16 + foldrms_row)
                if foldrms_row == 0:
                    for foldrms_step in range_constexpr(11):
                        foldrms_index = foldrms_tid + foldrms_step * 256
                        if foldrms_index < 2752:
                            if foldrms_index >= 2560 or foldrms_index % 16 >= 8:
                                fx.ptr_store(fx.Uint8(127), foldrms_scales + foldrms_index)

            def fold_chunk_stage(fc_partial, fc_residual, fc_shifted_gamma, fc_normalized, fc_residual_out, fc_activation, fc_scales, fc_epoch):
                # owner b folds/normalizes columns [640 * (b // 8), +640) of row b % 8.
                (fc_row, fc_chunk, fc_tid) = (iteration_block % 8, iteration_block // 8, iteration_tid)
                (fc_lane, fc_wave) = (fc_tid % 64, fc_tid // 64)
                fc_partials = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                llvm.InlineAsmOp(None, [], 's_setreg_imm32_b32 0xdc1, 1;', '~{memory}', has_side_effects=True)
                # the chunk's 30 Down CTAs of the previous layer have published.
                if fc_tid == 0:
                    fc_target = layer * 30
                    fc_have = fx.Int32(0)
                    fc_bad = coherent_u32(native + 289)
                    fc_waits = fx.Int32(0)
                    while (fc_have < fc_target) & (fc_bad == 0) & (fc_waits < 1048576):
                        fc_have = coherent_u32(native + CHUNK_WORD + fc_chunk * 16)
                        fc_bad = coherent_u32(native + 289)
                        llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                        fc_waits = fc_waits + 1
                    fc_ans = fx.Int32(1)
                    if (fc_have < fc_target) | (fc_bad != 0):
                        fail_native()
                        fc_ans = fx.Int32(2)
                    llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)\nbuffer_inv sc1', '~{memory}', has_side_effects=True)
                    fx.ptr_store(fc_ans, status_ptr + 6)
                gpu.barrier()
                fc_chunk_ok = fx.Int32(fx.ptr_load(status_ptr + 6))
                fc_pd = nrm_desc(fc_partial, 491520)
                fc_rd = nrm_desc(fc_residual, 81920)
                fc_gd = nrm_desc(fc_shifted_gamma, 10240)
                fc_od = nrm_desc(fc_residual_out, 81920)
                fc_nd = nrm_desc(fc_normalized, 81920)
                fc_column = fc_chunk * 640 + fc_tid * 8
                fc_valid = fc_tid < 80
                fc_offset = fc_row * 5120 + fc_column
                fc_parts = [nrm_ld16(fc_pd, nrm_offset(fc_valid, (fc_plane * 40960 + fc_offset) * 4 + fc_half * 16), 1) for fc_plane in range_constexpr(3) for fc_half in range_constexpr(2)]
                fc_g = nrm_ld16(fc_gd, nrm_offset(fc_valid, fc_column * 2), 0)
                fc_s = nrm_ld16(fc_rd, nrm_offset(fc_valid, fc_offset * 2), 1)
                fc_planes = [fx.Vector.from_elements([fc_parts[p * 2][i] for i in range_constexpr(4)] + [fc_parts[p * 2 + 1][i] for i in range_constexpr(4)], fx.Int32).bitcast(fx.Float32) for p in range_constexpr(3)]
                fc_gamma = fc_g.bitcast(fx.BFloat16)
                fc_skip = fc_s.bitcast(fx.BFloat16)
                fc_data = (fc_planes[0] + fc_planes[1] + fc_planes[2]).to(fx.BFloat16)
                fc_summed = fc_data.to(fx.Float32) + fc_skip.to(fx.Float32)
                fc_acc = fx.Vector(fmath.fma(fc_summed, fc_summed, fx.Vector.filled(8, 0.0, fx.Float32)))
                fc_total = fc_acc[0] + fc_acc[1] + (fc_acc[2] + fc_acc[3]) + (fc_acc[4] + fc_acc[5] + (fc_acc[6] + fc_acc[7]))
                fc_total = fx.Float32(llvm.inline_asm(T.f32, [fc_total.ir_value()], 's_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n', '=v,0', has_side_effects=True))
                fc_total = fc_total + fc_total.shuffle_xor(16, 64)
                fc_total = fc_total + fc_total.shuffle_xor(32, 64)
                if fc_lane == 0:
                    fx.ptr_store(fc_total, fc_partials + fc_wave)
                gpu.barrier()
                if fc_tid < 64:
                    fc_local = fx.Float32(0.0)
                    if fc_lane < 4:
                        fc_local = fx.Float32(fx.ptr_load(fc_partials + fc_lane))
                    fc_local = fx.Float32(llvm.inline_asm(T.f32, [fc_local.ir_value()], 's_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n s_nop 1\n v_add_f32_dpp $0, $0, $0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n', '=v,0', has_side_effects=True))
                    if fc_tid == 0:
                        fc_word = fx.Vector.from_elements([fx.Int32(fc_epoch), fc_local.bitcast(fx.Int32)], fx.Int32)
                        llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(native + PEER_WORD + iteration_block * 16)).ir_value(), fc_word.ir_value()], 'global_store_dwordx2 $0, $1, off sc0 sc1\ns_waitcnt vmcnt(0)', 'v,v,~{memory}', has_side_effects=True)
                    # Lane c < 8 polls chunk c's word of this row; all eight must carry the epoch.
                    fc_slot = native + PEER_WORD + (fc_lane % 8 * 8 + fc_row) * 16
                    fc_failed = fx.Int32(rocdl.readfirstlane(T.i32, coherent_u32(native + 289).ir_value()))
                    fc_ready = fx.Int32(0)
                    fc_value = fx.Int32(0)
                    fc_polls = fx.Int32(0)
                    while (fc_ready == 0) & (fc_failed == 0) & (fc_polls < 1048576):
                        fc_got = fx.Vector(llvm.inline_asm(T.vec(2, T.i32), [fx.Uint64(fx.ptrtoint(fc_slot)).ir_value()], 'global_load_dwordx2 $0, $1, off sc0 sc1\ns_waitcnt vmcnt(0)', '=&v,v,~{memory}', has_side_effects=True))
                        fc_ok = (fc_got[0] == fc_epoch).to(fx.Int32)
                        fc_value = fc_got[1]
                        for fc_shift in (1, 2, 4):
                            fc_ok = fx.min(fc_ok, fc_ok.shuffle_xor(fc_shift, 64))
                        fc_ready = fx.Int32(rocdl.readfirstlane(T.i32, fc_ok.ir_value()))
                        fc_failed = fx.Int32(rocdl.readfirstlane(T.i32, coherent_u32(native + 289).ir_value()))
                        llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                        fc_polls = fc_polls + 1
                    fc_parts8 = [fx.Int32(gpu.shuffle_idx(fc_value, fc_c, 64)).bitcast(fx.Float32) for fc_c in range_constexpr(8)]
                    fc_sum = fx.Float32(0.0)
                    for fc_c in range_constexpr(8):
                        fc_sum = fc_sum + fc_parts8[fc_c]
                    if fc_tid == 0:
                        fc_answer = fx.Int32(1)
                        if (fc_ready == 0) | (fc_failed != 0):
                            fail_native()
                            fc_answer = fx.Int32(2)
                        fx.ptr_store(fc_answer, status_ptr + 5)
                        fx.ptr_store(fc_sum, fc_partials + 8)
                gpu.barrier()
                fc_row_total = fx.Float32(fx.ptr_load(fc_partials + 8))
                fc_variance_eps = fx.Float32(fmath.fma(fc_row_total, fx.Float32(1.0 / 5120.0), fx.Float32(1e-06)))
                fc_rms = fx.Float32(rocdl.rsq(T.f32, fc_variance_eps.ir_value()))
                fc_rounded_sum = (fc_data.to(fx.Float32) + fc_skip.to(fx.Float32)).to(fx.BFloat16)
                fc_norm = (fc_rounded_sum.to(fx.Float32) * fc_rms * fc_gamma.to(fx.Float32)).to(fx.BFloat16)
                fc_store_offset = nrm_offset(fc_valid, fc_offset * 2)
                nrm_st16(fc_od, fc_store_offset, fc_rounded_sum.bitcast(fx.Int32), 3)
                nrm_st16(fc_nd, fc_store_offset, fc_norm.bitcast(fx.Int32), 3)
                fc_raw = fc_norm.bitcast(fx.Int32)
                fc_values = fc_norm.to(fx.Float32)
                fc_amax = fx.Float32(0.0)
                for fc_element in range_constexpr(8):
                    fc_amax = fx.max(fc_amax, fmath.absf(fc_values[fc_element]))
                for fc_shift in (1, 2):
                    fc_amax = fx.max(fc_amax, fc_amax.shuffle_xor(fc_shift, 64))
                fc_bits = (fc_amax * fx.Float32(1.0 / 448.0)).bitcast(fx.Int32)
                fc_exponent = (fc_bits.shrui(fx.Int32(23)) & fx.Int32(255)) + ((fc_bits & fx.Int32(8388607)) != 0).to(fx.Int32)
                fc_exponent = fx.max(fx.Int32(1), fx.min(fc_exponent, fx.Int32(254)))
                fc_hw_scale = (fc_exponent << fx.Int32(23)).bitcast(fx.Float32)
                fc_qw0 = fx.Int32(llvm.inline_asm(T.i32, [fc_raw[0].ir_value(), fc_hw_scale.ir_value(), fx.Int32(0).ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2', '=v,v,v,0', has_side_effects=True))
                fc_qw0 = fx.Int32(llvm.inline_asm(T.i32, [fc_raw[1].ir_value(), fc_hw_scale.ir_value(), fc_qw0.ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2 op_sel:[0,0,1]', '=v,v,v,0', has_side_effects=True))
                fc_qw1 = fx.Int32(llvm.inline_asm(T.i32, [fc_raw[2].ir_value(), fc_hw_scale.ir_value(), fx.Int32(0).ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2', '=v,v,v,0', has_side_effects=True))
                fc_qw1 = fx.Int32(llvm.inline_asm(T.i32, [fc_raw[3].ir_value(), fc_hw_scale.ir_value(), fc_qw1.ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2 op_sel:[0,0,1]', '=v,v,v,0', has_side_effects=True))
                if fc_valid:
                    fc_output = fx.Vector.from_elements([fc_qw0, fc_qw1], fx.Int32)
                    llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(fc_activation + fc_offset)).ir_value(), fc_output.ir_value()], 'global_store_dwordx2 $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                    if fc_tid % 4 == 0:
                        fc_qb = fc_column // 32
                        llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(fc_scales + fc_qb // 4 * 64 + fc_qb % 4 * 16 + fc_row)).ir_value(), fx.Int32(fc_exponent).ir_value()], 'global_store_byte $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                if iteration_block == 0:
                    for fc_step in range_constexpr(11):
                        fc_index = fc_tid + fc_step * 256
                        if fc_index < 2752:
                            if fc_index >= 2560 or fc_index % 16 >= 8:
                                llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(fc_scales + fc_index)).ir_value(), fx.Int32(127).ir_value()], 'global_store_byte $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                return fx.max(fc_chunk_ok, fx.Int32(fx.ptr_load(status_ptr + 5)))

            def fold_ready_wait(status_slot):
                # every fold counter has its 1 + 8 L publications.
                if iteration_tid < 64:
                    fr_target = layer * 8 + 1
                    fr_ready = fx.Int32(0)
                    fr_failed = fx.Int32(rocdl.readfirstlane(T.i32, coherent_u32(native + 289).ir_value()))
                    fr_polls = fx.Int32(0)
                    while (fr_ready == 0) & (fr_failed == 0) & (fr_polls < 1048576):
                        fr_have = coherent_u32(native + CHUNK_WORD + 512 + iteration_tid % 8 * 16)
                        fr_ok = (fr_have >= fr_target).to(fx.Int32)
                        for fr_shift in (1, 2, 4):
                            fr_ok = fx.min(fr_ok, fr_ok.shuffle_xor(fr_shift, 64))
                        fr_ready = fx.Int32(rocdl.readfirstlane(T.i32, fr_ok.ir_value()))
                        fr_failed = fx.Int32(rocdl.readfirstlane(T.i32, coherent_u32(native + 289).ir_value()))
                        llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                        fr_polls = fr_polls + 1
                    if iteration_tid == 0:
                        fr_answer = fx.Int32(1)
                        if (fr_ready == 0) | (fr_failed != 0):
                            fail_native()
                            fr_answer = fx.Int32(2)
                        llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)\nbuffer_inv sc1', '~{memory}', has_side_effects=True)
                        fx.ptr_store(fr_answer, status_ptr + status_slot)
                gpu.barrier()
                return fx.Int32(fx.ptr_load(status_ptr + status_slot))

            def qkv_stage(qkv_a, qkv_a_scale, qkv_weight, qkv_weight_scale, qkv_conv_weight, qkv_history, qkv_raw_output, qkv_workspace, qkv_conv_snapshot, qkv_ba, epoch, status_slot):
                result = fx.Int32(2)
                qkv_tid = iteration_tid
                (qkv_lane, qkv_wave) = (qkv_tid % 64, qkv_tid // 64)
                qkv_tile = iteration_block
                if alive == 1:
                    qkv_reduction = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                    qkv_demangle = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(qkv_reduction))

                    def qkv_uniform(qkv_v):
                        return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(qkv_v).ir_value()))

                    def qkv_descriptor(qkv_p, qkv_size):
                        qkv_addr = fx.Int64(fx.ptrtoint(qkv_p))
                        return fx.Vector.from_elements([qkv_uniform(qkv_addr.to(fx.Int32)), qkv_uniform(qkv_addr.shrui(fx.Int64(32)).to(fx.Int32)), fx.Int32(qkv_size), fx.Int32(159744)], fx.Int32)

                    def qkv_raw16(qkv_desc, qkv_offset, qkv_aux):
                        return fx.Vector(llvm.call_intrinsic(T.vec(4, T.i32), 'llvm.amdgcn.raw.buffer.load.v4i32', [qkv_desc.ir_value(), fx.Int32(qkv_offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(qkv_aux).ir_value()], [], []))

                    def qkv_raw4(qkv_desc, qkv_offset):
                        return fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.raw.buffer.load.i32', [qkv_desc.ir_value(), fx.Int32(qkv_offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value()], [], []))

                    def qkv_pin4(qkv_v):
                        return fx.Vector(llvm.inline_asm(T.vec(4, T.i32), [qkv_v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))

                    def qkv_pin1(qkv_v):
                        return fx.Int32(llvm.inline_asm(T.i32, [qkv_v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))
                    qkv_ad = qkv_descriptor(qkv_a, 40960)
                    qkv_asd = qkv_descriptor(qkv_a_scale, 2752)
                    qkv_bd = qkv_descriptor(qkv_weight, 42270720)
                    qkv_bsd = qkv_descriptor(qkv_weight_scale, 2641920)
                    qkv_alo = [fx.Vector.filled(4, 0, fx.Int32) for qkv__ in range_constexpr(10)]
                    qkv_ahi = [fx.Vector.filled(4, 0, fx.Int32) for qkv__ in range_constexpr(10)]
                    qkv_asc = [fx.Int32(0) for qkv__ in range_constexpr(10)]
                    qkv_bv = [fx.Vector.filled(4, 0, fx.Int32) for qkv__ in range_constexpr(24)]
                    qkv_bs = [fx.Int32(0) for qkv__ in range_constexpr(24)]
                    qkv_acc = [fx.Vector.filled(4, 0.0, fx.Float32) for qkv__ in range_constexpr(4)]
                    # P = 6 k-steps primed before the fold wait.
                    for qkv_r in range_constexpr(6):
                        qkv_kb = qkv_r * 4 + qkv_wave
                        for qkv_atom in range_constexpr(4):
                            qkv_bv[qkv_r * 4 + qkv_atom] = qkv_raw16(qkv_bd, ((qkv_tile * 4 + qkv_atom) * 40 + qkv_kb) * 1024 + qkv_lane * 16, 2)
                            qkv_bs[qkv_r * 4 + qkv_atom] = qkv_raw4(qkv_bsd, (qkv_kb * 1032 + qkv_tile * 4 + qkv_atom) * 64 + (qkv_lane & fx.Int32(-4)))
                    # k-steps 6-9 into LDS (DMA) and their scales into VGPRs, before the fold wait.
                    qkv_ldsw = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(arena + 65536) + qkv_wave * 16384).ir_value()))
                    qkv_bs2 = [fx.Int32(0) for qkv__ in range_constexpr(16)]
                    for qkv_r2 in range_constexpr(4):
                        qkv_kb = (6 + qkv_r2) * 4 + qkv_wave
                        for qkv_atom in range_constexpr(4):
                            llvm.InlineAsmOp(None, [fx.Int32(rocdl.readfirstlane(T.i32, (qkv_ldsw + (qkv_r2 * 4 + qkv_atom) * 1024).ir_value())).ir_value(), fx.Int32(((qkv_tile * 4 + qkv_atom) * 40 + qkv_kb) * 1024 + qkv_lane * 16).ir_value(), qkv_bd.ir_value()], 's_mov_b32 m0, $0\ns_nop 0\nbuffer_load_dwordx4 $1, $2, 0 offen nt lds', 's,v,s,~{memory},~{m0}', has_side_effects=True)
                            qkv_bs2[qkv_r2 * 4 + qkv_atom] = qkv_raw4(qkv_bsd, (qkv_kb * 1032 + qkv_tile * 4 + qkv_atom) * 64 + (qkv_lane & fx.Int32(-4)))
                    result = fold_ready_wait(status_slot)
                    if result == 1:
                        # epilogue operands (BA k-block, conv weights/history) also leave now.
                        qkv_ba_on = (qkv_tile >= 176).to(fx.Int32)
                        qkv_ba_job = fx.max(qkv_tile - 176, fx.Int32(0))
                        (qkv_ba_tile_h, qkv_ba_peer_h) = (qkv_ba_job // 10, qkv_ba_job % 10)
                        qkv_ba_kb = qkv_ba_peer_h * 4 + qkv_wave
                        qkv_ba_aoff = qkv_lane % 16 * 5120 + qkv_ba_kb * 128 + qkv_lane // 16 * 16
                        qkv_oob = (1 - qkv_ba_on) * fx.Int32(0x10000000)
                        qkv_ba_lo = qkv_raw16(qkv_ad, qkv_ba_aoff + qkv_oob, 0)
                        qkv_ba_hi = qkv_raw16(qkv_ad, qkv_ba_aoff + 64 + qkv_oob, 0)
                        qkv_ba_sa = qkv_raw4(qkv_asd, qkv_ba_kb * 64 + (qkv_lane & fx.Int32(-4)) + qkv_oob)
                        qkv_ba_b = qkv_raw16(qkv_bd, ((1024 + qkv_ba_tile_h) * 40 + qkv_ba_kb) * 1024 + qkv_lane * 16 + qkv_oob, 0)
                        qkv_ba_sb = qkv_raw4(qkv_bsd, (qkv_ba_kb * 1032 + 1024 + qkv_ba_tile_h) * 64 + (qkv_lane & fx.Int32(-4)) + qkv_oob)
                        qkv_conv_channel = fx.min(qkv_tile, fx.Int32(159)) * 64 + qkv_lane
                        qkv_conv_w = [fx.ptr_load(qkv_conv_weight + qkv_conv_channel * 4 + qkv_tap).to(fx.Float32) for qkv_tap in range_constexpr(4)]
                        qkv_conv_h = [fx.ptr_load(qkv_history + qkv_conv_channel * 4 + 1 + qkv_tap).to(fx.Float32) for qkv_tap in range_constexpr(3)]
                        # every activation k-block in one round trip.
                        for qkv_r in range_constexpr(10):
                            qkv_kb = qkv_r * 4 + qkv_wave
                            qkv_off = qkv_lane % 16 * 5120 + qkv_kb * 128 + qkv_lane // 16 * 16
                            qkv_alo[qkv_r] = qkv_raw16(qkv_ad, qkv_off, 0)
                            qkv_ahi[qkv_r] = qkv_raw16(qkv_ad, qkv_off + 64, 0)
                            qkv_asc[qkv_r] = qkv_raw4(qkv_asd, qkv_kb * 64 + (qkv_lane & fx.Int32(-4)))
                        for qkv_k in range_constexpr(10):
                            qkv_ar = qkv_k
                            qkv_br = qkv_k % 6
                            qkv_lo = qkv_pin4(qkv_alo[qkv_ar])
                            qkv_hi = qkv_pin4(qkv_ahi[qkv_ar])
                            qkv_sa = qkv_pin1(qkv_asc[qkv_ar])
                            qkv_sa = qkv_sa.shrui((qkv_lane & fx.Int32(3)) * 8) & fx.Int32(255)
                            qkv_av = fx.Vector.from_elements([qkv_lo[0], qkv_lo[1], qkv_lo[2], qkv_lo[3], qkv_hi[0], qkv_hi[1], qkv_hi[2], qkv_hi[3]], fx.Int32)
                            for qkv_atom in range_constexpr(4):
                                if fx.const_expr(qkv_k < 6):
                                    qkv_b = qkv_pin4(qkv_bv[qkv_br * 4 + qkv_atom])
                                    qkv_sb = qkv_pin1(qkv_bs[qkv_br * 4 + qkv_atom])
                                else:
                                    # from LDS (the DMA landed before the activation wait).
                                    qkv_b = fx.Vector(fx.ptr_load(arena + 65536 + qkv_wave * 16384 + ((qkv_k - 6) * 4 + qkv_atom) * 1024 + qkv_lane * 16, result_type=T.vec(16, T.i8))).bitcast(fx.Int32)
                                    qkv_sb = qkv_pin1(qkv_bs2[(qkv_k - 6) * 4 + qkv_atom])
                                qkv_sb = qkv_sb.shrui((qkv_lane & fx.Int32(3)) * 8) & fx.Int32(255)
                                qkv_bf = fx.Vector.from_elements([qkv_b[0], qkv_b[1], qkv_b[2], qkv_b[3], fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0)], fx.Int32)
                                qkv_acc[qkv_atom] = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [qkv_av, qkv_bf, qkv_acc[qkv_atom], 0, 4, 0, qkv_sa, 0, qkv_sb]))
                            if fx.const_expr(qkv_k + 6 < 6):
                                qkv_kb = (qkv_k + 6) * 4 + qkv_wave
                                for qkv_atom in range_constexpr(4):
                                    qkv_bv[qkv_br * 4 + qkv_atom] = qkv_raw16(qkv_bd, ((qkv_tile * 4 + qkv_atom) * 40 + qkv_kb) * 1024 + qkv_lane * 16, 2)
                                    qkv_bs[qkv_br * 4 + qkv_atom] = qkv_raw4(qkv_bsd, (qkv_kb * 1032 + qkv_tile * 4 + qkv_atom) * 64 + (qkv_lane & fx.Int32(-4)))
                        for qkv_atom in range_constexpr(4):
                            for qkv_e in range_constexpr(4):
                                fx.ptr_store(qkv_acc[qkv_atom][qkv_e], qkv_reduction + qkv_atom * 1024 + qkv_wave * 256 + qkv_lane * 4 + qkv_e)
                        gpu.barrier()
                        if qkv_wave == 0:
                            for qkv_atom in range_constexpr(4):
                                qkv_folded = [fx.Float32(0.0) for qkv__ in range_constexpr(4)]
                                for qkv_e in range_constexpr(4):
                                    qkv_off = qkv_atom * 1024 + qkv_lane * 4 + qkv_e
                                    qkv_value = fx.ptr_load(qkv_reduction + qkv_off)
                                    for qkv_w in range_constexpr(1, 4):
                                        qkv_value = qkv_value + fx.ptr_load(qkv_reduction + qkv_off + qkv_w * 256)
                                    qkv_folded[qkv_e] = qkv_value
                                qkv_rounded = fx.Vector.from_elements(qkv_folded, fx.Float32).to(fx.BFloat16)
                                for qkv_e in range_constexpr(4):
                                    fx.ptr_store(qkv_rounded[qkv_e], qkv_demangle + qkv_atom * 320 + (qkv_lane // 16 * 4 + qkv_e) * 20 + qkv_lane % 16)
                        gpu.barrier()
                        if qkv_wave == 0:
                            qkv_row = qkv_lane // 4
                            qkv_col = qkv_lane % 4 * 4
                            if qkv_row < 8:
                                for qkv_atom in range_constexpr(4):
                                    qkv_values = fx.ptr_load(qkv_demangle + qkv_atom * 320 + qkv_row * 20 + qkv_col, result_type=T.vec(4, T.bf16))
                                    # write-through (point-to-point consumers).
                                    llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(qkv_raw_output + qkv_row * 16512 + qkv_tile * 64 + qkv_atom * 16 + qkv_col)).ir_value(), qkv_values.bitcast(fx.Int32).ir_value()], 'global_store_dwordx2 $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                        gpu.barrier()
                        if qkv_tile < 160:
                            qkv_channel = qkv_tile * 64 + qkv_lane
                            qkv_weights = [fx.Float32(0.0) for qkv__ in range_constexpr(4)]
                            for qkv_tap in range_constexpr(4):
                                qkv_weights[qkv_tap] = qkv_conv_w[qkv_tap]
                            for qkv_pair in range_constexpr(2):
                                qkv_row = qkv_wave * 2 + qkv_pair
                                qkv_value = fx.Float32(0.0)
                                for qkv_tap in range_constexpr(4):
                                    qkv_source = qkv_row - 3 + qkv_tap
                                    qkv_x = fx.Float32(0.0)
                                    if qkv_source < 0:
                                        qkv_x = qkv_conv_h[2]
                                        if qkv_source == -3:
                                            qkv_x = qkv_conv_h[0]
                                        if qkv_source == -2:
                                            qkv_x = qkv_conv_h[1]
                                    else:
                                        qkv_x = fx.ptr_load(qkv_demangle + qkv_lane // 16 * 320 + qkv_source * 20 + qkv_lane % 16).to(fx.Float32)
                                    qkv_value = fx.Float32(llvm.call_intrinsic(T.f32, 'llvm.fma.f32', [qkv_x.ir_value(), qkv_weights[qkv_tap].ir_value(), qkv_value.ir_value()], [], []))
                                qkv_exponent = -qkv_value * fx.Float32(1.4426950408889634)
                                qkv_exponential = fx.Float32(llvm.inline_asm(T.f32, [qkv_exponent.ir_value()], 'v_exp_f32 $0, $1', '=v,v', has_side_effects=False))
                                qkv_denominator = fx.Float32(1.0) + qkv_exponential
                                qkv_inverse = fx.Float32(llvm.inline_asm(T.f32, [qkv_denominator.ir_value()], 'v_rcp_f32 $0, $1', '=v,v', has_side_effects=False))
                                qkv_ws = fx.Vector.from_elements([(qkv_value * qkv_inverse).to(fx.BFloat16), fx.BFloat16(0.0)], fx.BFloat16).bitcast(fx.Int32)[0]
                                llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(qkv_workspace + qkv_row * 16512 + qkv_channel)).ir_value(), qkv_ws.ir_value()], 'global_store_short $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                            qkv_saved = fx.ptr_load(qkv_demangle + qkv_lane // 16 * 320 + (4 + qkv_wave) * 20 + qkv_lane % 16)
                            fx.ptr_store(qkv_saved, qkv_conv_snapshot + qkv_channel * 4 + qkv_wave)
                        gpu.barrier()
                        if qkv_tile >= 176:
                            qkv_job = qkv_tile - 176
                            (qkv_ba_tile, qkv_peer) = (qkv_job // 10, qkv_job % 10)
                            qkv_kb = qkv_peer * 4 + qkv_wave
                            qkv_off = qkv_lane % 16 * 5120 + qkv_kb * 128 + qkv_lane // 16 * 16
                            (qkv_lo, qkv_hi, qkv_sa, qkv_b, qkv_sb) = (qkv_ba_lo, qkv_ba_hi, qkv_ba_sa, qkv_ba_b, qkv_ba_sb)
                            qkv_sa = qkv_sa.shrui((qkv_lane & fx.Int32(3)) * 8) & fx.Int32(255)
                            qkv_sb = qkv_sb.shrui((qkv_lane & fx.Int32(3)) * 8) & fx.Int32(255)
                            qkv_av = fx.Vector.from_elements([qkv_lo[0], qkv_lo[1], qkv_lo[2], qkv_lo[3], qkv_hi[0], qkv_hi[1], qkv_hi[2], qkv_hi[3]], fx.Int32)
                            qkv_bf = fx.Vector.from_elements([qkv_b[0], qkv_b[1], qkv_b[2], qkv_b[3], fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0)], fx.Int32)
                            qkv_ba_acc = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [qkv_av, qkv_bf, fx.Vector.filled(4, 0.0, fx.Float32), 0, 4, 0, qkv_sa, 0, qkv_sb]))
                            for qkv_e in range_constexpr(4):
                                fx.ptr_store(qkv_ba_acc[qkv_e], qkv_reduction + qkv_wave * 256 + qkv_lane * 4 + qkv_e)
                            gpu.barrier()
                            if qkv_wave == 0:
                                qkv_ba_folded = [fx.Float32(0.0) for qkv__ in range_constexpr(4)]
                                for qkv_e in range_constexpr(4):
                                    qkv_off = qkv_lane * 4 + qkv_e
                                    qkv_value = fx.ptr_load(qkv_reduction + qkv_off)
                                    for qkv_w in range_constexpr(1, 4):
                                        qkv_value = qkv_value + fx.ptr_load(qkv_reduction + qkv_off + qkv_w * 256)
                                    qkv_ba_folded[qkv_e] = qkv_value
                                for qkv_e in range_constexpr(4):
                                    qkv_row = qkv_lane // 16 * 4 + qkv_e
                                    fx.ptr_store(qkv_ba_folded[qkv_e], qkv_reduction + qkv_row * 20 + qkv_lane % 16)
                            gpu.barrier()
                            if qkv_wave == 0:
                                qkv_row = qkv_lane // 4
                                qkv_col = qkv_lane % 4 * 4
                                if qkv_row < 8:
                                    for qkv_e in range_constexpr(4):
                                        qkv_value = fx.ptr_load(qkv_reduction + qkv_row * 20 + qkv_col + qkv_e)
                                        fx.atomic_add(qkv_ba + qkv_row * 128 + qkv_ba_tile * 16 + qkv_col + qkv_e, qkv_value, ordering=fx.AtomicOrdering.Monotonic, syncscope='agent')
                            gpu.barrier()
                        # outputs drained -> head counters (key head / value head + z / BA tile).
                        llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                        gpu.barrier()
                        if qkv_tid == 0:
                            qkv_hc = fx.Int32(0)
                            if qkv_tile < 64:
                                qkv_hc = 2048 + qkv_tile % 32 // 2 * 16
                            else:
                                qkv_hc = 2304 + (qkv_tile - 64 - (qkv_tile >= 160).to(fx.Int32) * 96) // 2 * 16
                            system_atomic_add_nw(native + CHUNK_WORD + qkv_hc)
                            if qkv_tile >= 176:
                                system_atomic_add_nw(native + CHUNK_WORD + 3072 + (qkv_tile - 176) // 10 * 16)
                else:
                    result = alive
                return result

            def core_stage(core_raw_output, core_workspace, core_ba, core_initial_state, core_a_log, core_dt_bias, core_core, core_final_state, core_squares, core_state_snapshots):
                # point-to-point QKV -> core: this CTA's head counters.
                if iteration_tid < 64:
                    cw_d = layer - layer // 4 + 1
                    cw_failed = fx.Int32(rocdl.readfirstlane(T.i32, coherent_u32(native + 289).ir_value()))
                    cw_ready = fx.Int32(0)
                    cw_polls = fx.Int32(0)
                    cw_lane = iteration_tid
                    while (cw_ready == 0) & (cw_failed == 0) & (cw_polls < 1048576):
                        cw_off = fx.Int32(0)
                        cw_need = fx.Int32(0)
                        if cw_lane == 0:
                            cw_off = 2048 + iteration_block // 4 // 3 * 16
                            cw_need = cw_d * 4
                        if cw_lane == 1:
                            cw_off = 2304 + iteration_block // 4 * 16
                            cw_need = cw_d * 4
                        if cw_lane == 2:
                            cw_off = 3072 + iteration_block // 4 // 16 * 16
                            cw_need = cw_d * 10
                        if cw_lane == 3:
                            cw_off = 3072 + (48 + iteration_block // 4) // 16 * 16
                            cw_need = cw_d * 10
                        cw_have = coherent_u32(native + CHUNK_WORD + cw_off)
                        cw_ok = (cw_have >= cw_need).to(fx.Int32)
                        for cw_shift in (1, 2, 4, 8, 16, 32):
                            cw_ok = fx.min(cw_ok, cw_ok.shuffle_xor(cw_shift, 64))
                        cw_ready = fx.Int32(rocdl.readfirstlane(T.i32, cw_ok.ir_value()))
                        llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                        cw_polls = cw_polls + 1
                    if iteration_tid == 0:
                        cw_answer = fx.Int32(1)
                        if (cw_ready == 0) | (cw_failed != 0):
                            fail_native()
                            cw_answer = fx.Int32(2)
                        llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)\nbuffer_inv sc1', '~{memory}', has_side_effects=True)
                        fx.ptr_store(cw_answer, status_ptr + 2)
                gpu.barrier()
                core_tid = iteration_tid
                core_block = iteration_block
                (core_head, core_split) = (core_block // 4, core_block % 4)
                core_key_head = core_head // 3
                (core_lane, core_subgroup) = (core_tid % 16, core_tid // 16)
                (core_wave, core_wave_lane) = (core_tid // 64, core_tid % 64)
                core_value_base = core_split * 32 + core_subgroup * 2
                core_query = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                core_key = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 4096))
                core_value = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 8192))
                core_decay = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 8704))
                core_beta = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 8736))
                core_local_core = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 8768))

                def core_sum16(core_x):
                    # row_shl 8/4/2/1 adds as compiler-visible DPP (0x100 + n), zero fill.
                    for core_shl in (8, 4, 2, 1):
                        core_x = core_x + fx.Float32(rocdl.update_dpp(T.f32, fx.Float32(0.0).ir_value(), core_x.ir_value(), 0x100 + core_shl, 0xF, 0xF, True))
                    return core_x

                def core_native_rsqrt(core_x):
                    return fx.Float32(llvm.inline_asm(T.f32, [core_x.ir_value()], 'v_rsq_f32 $0, $1', '=v,v', has_side_effects=False))

                def core_exp2_fast(core_x):
                    return fx.Float32(llvm.inline_asm(T.f32, [core_x.ir_value()], 'v_exp_f32 $0, $1', '=v,v', has_side_effects=False))

                def core_sigmoid(core_x):
                    core_denominator = fx.Float32(1.0) + core_exp2_fast(-core_x * fx.Float32(1.4426950408889634))
                    return fx.Float32(llvm.inline_asm(T.f32, [core_denominator.ir_value()], 'v_rcp_f32 $0, $1', '=v,v', has_side_effects=False))
                core_state0 = [fx.Float32(0.0) for core__ in range_constexpr(8)]
                core_state1 = [fx.Float32(0.0) for core__ in range_constexpr(8)]
                core_state_base = core_head * 16384 + core_value_base * 128 + core_lane
                # issue every prologue load before any arithmetic.
                core_pre = []
                for core_pair in range_constexpr(2):
                    core_row = core_wave + core_pair * 4
                    core_packed = core_row * 16512
                    core_pre.append((fx.ptr_load(core_workspace + core_packed + core_key_head * 128 + core_wave_lane),
                                     fx.ptr_load(core_workspace + core_packed + core_key_head * 128 + core_wave_lane + 64),
                                     fx.ptr_load(core_workspace + core_packed + 2048 + core_key_head * 128 + core_wave_lane),
                                     fx.ptr_load(core_workspace + core_packed + 2048 + core_key_head * 128 + core_wave_lane + 64),
                                     fx.ptr_load(core_workspace + core_packed + 4096 + core_head * 128 + core_split * 32 + core_wave_lane % 32),
                                     fx.ptr_load(core_ba + core_row * 128 + core_head),
                                     fx.ptr_load(core_ba + core_row * 128 + 48 + core_head)))
                core_dt = fx.ptr_load(core_dt_bias + core_head)
                core_alog = fx.ptr_load(core_a_log + core_head)
                # recurrent state after the prologue loads (needed only by the recurrence).
                for core_index in range_constexpr(8):
                    core_state0[core_index] = fx.ptr_load(core_initial_state + core_state_base + core_index * 16)
                    core_state1[core_index] = fx.ptr_load(core_initial_state + core_state_base + 128 + core_index * 16)
                for core_pair in range_constexpr(2):
                    core_row = core_wave + core_pair * 4
                    core_q0 = core_pre[core_pair][0].to(fx.Float32)
                    core_q1 = core_pre[core_pair][1].to(fx.Float32)
                    core_k0 = core_pre[core_pair][2].to(fx.Float32)
                    core_k1 = core_pre[core_pair][3].to(fx.Float32)
                    core_qs = fmath.fma(core_q0, core_q0, core_q1 * core_q1)
                    core_ks = fmath.fma(core_k0, core_k0, core_k1 * core_k1)
                    for core_shift in (32, 16):
                        core_qs = core_qs + gpu.shuffle_down(core_qs, core_shift, 64)
                        core_ks = core_ks + gpu.shuffle_down(core_ks, core_shift, 64)
                    core_qs = core_sum16(core_qs)
                    core_ks = core_sum16(core_ks)
                    core_qi = core_native_rsqrt(gpu.shuffle_idx(core_qs, 0, 64) + fx.Float32(1e-06)) * core_native_rsqrt(fx.Float32(128.0))
                    core_ki = core_native_rsqrt(gpu.shuffle_idx(core_ks, 0, 64) + fx.Float32(1e-06))
                    fx.ptr_store(core_q0 * core_qi, core_query + core_row * 128 + core_wave_lane)
                    fx.ptr_store(core_q1 * core_qi, core_query + core_row * 128 + core_wave_lane + 64)
                    fx.ptr_store(core_k0 * core_ki, core_key + core_row * 128 + core_wave_lane)
                    fx.ptr_store(core_k1 * core_ki, core_key + core_row * 128 + core_wave_lane + 64)
                    core_v = core_pre[core_pair][4]
                    if core_wave_lane < 32:
                        fx.ptr_store(core_v, core_value + core_row * 32 + core_wave_lane)
                    core_bv = core_pre[core_pair][5].to(fx.BFloat16).to(fx.Float32)
                    core_av = core_pre[core_pair][6].to(fx.BFloat16).to(fx.Float32) + core_dt
                    if core_wave_lane == 0:
                        core_softplus = core_av
                        if core_av <= fx.Float32(20.0):
                            core_arg = fx.Float32(1.0) + core_exp2_fast(core_av * fx.Float32(1.4426950408889634))
                            core_lg = fx.Float32(llvm.inline_asm(T.f32, [core_arg.ir_value()], 'v_log_f32 $0, $1', '=v,v', has_side_effects=False))
                            core_softplus = core_lg * fx.Float32(0.6931471805599453)
                        core_log_decay = -fmath.exp(core_alog) * core_softplus
                        fx.ptr_store(fmath.exp(core_log_decay), core_decay + core_row)
                        fx.ptr_store(core_sigmoid(core_bv), core_beta + core_row)
                gpu.barrier()
                core_result0 = fx.Float32(0.0)
                core_result1 = fx.Float32(0.0)
                for core_row in range_constexpr(8):
                    core_keys = [fx.Float32(0.0) for core__ in range_constexpr(8)]
                    core_pred0 = fx.Float32(0.0)
                    core_pred1 = fx.Float32(0.0)
                    core_d = fx.ptr_load(core_decay + core_row)
                    for core_index in range_constexpr(8):
                        core_keys[core_index] = fx.ptr_load(core_key + core_row * 128 + core_lane + core_index * 16)
                        core_state0[core_index] = core_state0[core_index] * core_d
                        core_state1[core_index] = core_state1[core_index] * core_d
                        core_pred0 = fmath.fma(core_state0[core_index], core_keys[core_index], core_pred0)
                        core_pred1 = fmath.fma(core_state1[core_index], core_keys[core_index], core_pred1)
                    core_pred0 = core_sum16(core_pred0)
                    core_pred1 = core_sum16(core_pred1)
                    if fx.const_expr(core_row > 0):
                        core_result0 = core_sum16(core_result0)
                        core_result1 = core_sum16(core_result1)
                        if core_lane == 0:
                            fx.ptr_store(core_result0, core_core + (core_row - 1) * 6144 + core_head * 128 + core_value_base)
                            fx.ptr_store(core_result1, core_core + (core_row - 1) * 6144 + core_head * 128 + core_value_base + 1)
                            fx.ptr_store(core_result0, core_local_core + (core_row - 1) * 32 + core_subgroup * 2)
                            fx.ptr_store(core_result1, core_local_core + (core_row - 1) * 32 + core_subgroup * 2 + 1)
                    core_delta0 = fx.Float32(0.0)
                    core_delta1 = fx.Float32(0.0)
                    if core_lane == 0:
                        core_b = fx.ptr_load(core_beta + core_row)
                        core_delta0 = core_b * (fx.ptr_load(core_value + core_row * 32 + core_subgroup * 2).to(fx.Float32) - core_pred0)
                        core_delta1 = core_b * (fx.ptr_load(core_value + core_row * 32 + core_subgroup * 2 + 1).to(fx.Float32) - core_pred1)
                    core_delta0 = gpu.shuffle_idx(core_delta0, core_wave_lane - core_lane, 64)
                    core_delta1 = gpu.shuffle_idx(core_delta1, core_wave_lane - core_lane, 64)
                    core_result0 = fx.Float32(0.0)
                    core_result1 = fx.Float32(0.0)
                    for core_index in range_constexpr(8):
                        core_state0[core_index] = fmath.fma(core_delta0, core_keys[core_index], core_state0[core_index])
                        core_state1[core_index] = fmath.fma(core_delta1, core_keys[core_index], core_state1[core_index])
                        core_q = fx.ptr_load(core_query + core_row * 128 + core_lane + core_index * 16)
                        core_result0 = fmath.fma(core_state0[core_index], core_q, core_result0)
                        core_result1 = fmath.fma(core_state1[core_index], core_q, core_result1)
                        if fx.const_expr(diagnostics):
                            fx.ptr_store(core_state0[core_index], core_state_snapshots + core_row * 786432 + core_state_base + core_index * 16)
                            fx.ptr_store(core_state1[core_index], core_state_snapshots + core_row * 786432 + core_state_base + 128 + core_index * 16)
                core_result0 = core_sum16(core_result0)
                core_result1 = core_sum16(core_result1)
                if core_lane == 0:
                    fx.ptr_store(core_result0, core_core + 7 * 6144 + core_head * 128 + core_value_base)
                    fx.ptr_store(core_result1, core_core + 7 * 6144 + core_head * 128 + core_value_base + 1)
                    fx.ptr_store(core_result0, core_local_core + 7 * 32 + core_subgroup * 2)
                    fx.ptr_store(core_result1, core_local_core + 7 * 32 + core_subgroup * 2 + 1)
                for core_index in range_constexpr(8):
                    fx.ptr_store(core_state0[core_index], core_final_state + core_state_base + core_index * 16)
                    fx.ptr_store(core_state1[core_index], core_final_state + core_state_base + 128 + core_index * 16)
                gpu.barrier()
                core_row = core_tid // 32
                core_column = core_tid % 32
                core_c = fx.ptr_load(core_local_core + core_row * 32 + core_column)
                core_square = fmath.fma(core_c, core_c, gpu.shuffle_down(core_c * core_c, 16, 32))
                for core_shift in (8, 4, 2, 1):
                    core_square = core_square + gpu.shuffle_down(core_square, core_shift, 32)
                if core_column == 0:
                    llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(native + PEER_WORD + (core_head * 4 + core_split) * 16 + core_row)).ir_value(), fx.Int32(core_square.bitcast(fx.Int32)).ir_value()], 'global_store_dword $0, $1, off sc0 sc1\ns_waitcnt vmcnt(0)', 'v,v,~{memory}', has_side_effects=True)
                    if fx.const_expr(diagnostics):
                        fx.ptr_store(core_square, core_squares + core_head * 32 + core_split * 8 + core_row)

            def core_tail_stage(core_raw_output, core_workspace, core_ba):
                # point-to-point QKV -> core: this CTA's head counters.
                if iteration_tid < 64:
                    ct_d = layer - layer // 4 + 1
                    ct_failed = fx.Int32(rocdl.readfirstlane(T.i32, coherent_u32(native + 289).ir_value()))
                    ct_ready = fx.Int32(0)
                    ct_polls = fx.Int32(0)
                    ct_lane = iteration_tid
                    while (ct_ready == 0) & (ct_failed == 0) & (ct_polls < 1048576):
                        ct_off = fx.Int32(2304) + fx.min(ct_lane, fx.Int32(47)) * 16
                        ct_need = ct_d * 4
                        if ct_lane >= 48:
                            ct_off = 3072 + fx.min(ct_lane - 48, fx.Int32(7)) * 16
                            ct_need = ct_d * 10
                        if ct_lane >= 56:
                            ct_need = fx.Int32(0)
                        ct_have = coherent_u32(native + CHUNK_WORD + ct_off)
                        ct_ok = (ct_have >= ct_need).to(fx.Int32)
                        for ct_shift in (1, 2, 4, 8, 16, 32):
                            ct_ok = fx.min(ct_ok, ct_ok.shuffle_xor(ct_shift, 64))
                        ct_ready = fx.Int32(rocdl.readfirstlane(T.i32, ct_ok.ir_value()))
                        llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                        ct_polls = ct_polls + 1
                    if iteration_tid == 0:
                        ct_answer = fx.Int32(1)
                        if (ct_ready == 0) | (ct_failed != 0):
                            fail_native()
                            ct_answer = fx.Int32(2)
                        llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)\nbuffer_inv sc1', '~{memory}', has_side_effects=True)
                        fx.ptr_store(ct_answer, status_ptr + 2)
                gpu.barrier()
                # payload tail on CTAs 192-255 (nothing in the forward reads it).
                core_block = iteration_block - 192
                core_tid = iteration_tid
                for core_tail_pass in range_constexpr(4):
                    core_tail_index = core_block * 256 + core_tid + core_tail_pass * (64 * 256)
                    if core_tail_index < 8 * 6272:
                        core_tail_row = core_tail_index // 6272
                        core_tail_column = core_tail_index % 6272
                        core_tail_value = fx.BFloat16(0.0)
                        if core_tail_column < 6144:
                            core_tail_value = fx.ptr_load(core_raw_output + core_tail_row * 16512 + 10240 + core_tail_column)
                        else:
                            core_tail_value = fx.ptr_load(core_ba + core_tail_row * 128 + core_tail_column - 6144).to(fx.BFloat16)
                            fx.ptr_store(core_tail_value, core_raw_output + core_tail_row * 16512 + 10240 + core_tail_column)
                        fx.ptr_store(core_tail_value, core_workspace + core_tail_row * 16512 + 10240 + core_tail_column)

            def finalize_stage(finalize_z_input, finalize_core, finalize_squares, finalize_norm, finalize_output, finalize_activation, finalize_scales):
                finalize_tid = iteration_tid
                finalize_block = iteration_block
                (finalize_head, finalize_split) = (finalize_block // 4, finalize_block % 4)
                (finalize_row, finalize_lane) = (finalize_tid // 32, finalize_tid % 32)
                finalize_local_output = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                finalize_inverse_ptr = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 8704))
                finalize_inverse = fx.Float32(fx.ptr_load(finalize_inverse_ptr + finalize_row))
                finalize_column = finalize_head * 128 + finalize_split * 32 + finalize_lane
                finalize_z = fx.ptr_load(finalize_z_input + finalize_row * 16512 + 10240 + finalize_column).to(fx.Float32)
                finalize_arg = -finalize_z * fx.Float32(1.4426950408889634)
                finalize_exp = fx.Float32(llvm.inline_asm(T.f32, [finalize_arg.ir_value()], 'v_exp_f32 $0, $1', '=v,v', has_side_effects=False))
                finalize_den = fx.Float32(1.0) + finalize_exp
                finalize_sig = fx.Float32(llvm.inline_asm(T.f32, [finalize_den.ir_value()], 'v_rcp_f32 $0, $1', '=v,v', has_side_effects=False))
                finalize_v = fx.ptr_load(finalize_core + finalize_row * 6144 + finalize_column) * finalize_inverse
                finalize_v = finalize_v * fx.ptr_load(finalize_norm + finalize_split * 32 + finalize_lane).to(fx.Float32)
                finalize_v = finalize_v * finalize_z
                finalize_v = finalize_v * finalize_sig
                finalize_rounded = finalize_v.to(fx.BFloat16)
                fx.ptr_store(finalize_rounded, finalize_output + finalize_row * 6144 + finalize_column)
                fx.ptr_store(finalize_rounded, finalize_local_output + finalize_tid)
                gpu.barrier()
                llvm.InlineAsmOp(None, [], 's_setreg_imm32_b32 0xdc1, 1', '~{memory}', has_side_effects=True)
                if finalize_lane < 4:
                    finalize_raw = fx.Vector(fx.ptr_load(finalize_local_output + finalize_row * 32 + finalize_lane * 8, result_type=T.vec(8, T.bf16))).bitcast(fx.Int32)
                    finalize_values = finalize_raw.bitcast(fx.BFloat16).to(fx.Float32)
                    finalize_amax = fx.Float32(0.0)
                    for finalize_element in range_constexpr(8):
                        finalize_amax = fx.max(finalize_amax, fmath.absf(finalize_values[finalize_element]))
                    for finalize_shift in (1, 2):
                        finalize_amax = fx.max(finalize_amax, finalize_amax.shuffle_xor(finalize_shift, 64))
                    finalize_ratio = finalize_amax * fx.Float32(1.0 / 448.0)
                    finalize_bits = finalize_ratio.bitcast(fx.Int32)
                    finalize_exponent = finalize_bits.shrui(fx.Int32(23)) & fx.Int32(255)
                    if finalize_bits & fx.Int32(8388607) != 0:
                        finalize_exponent = finalize_exponent + 1
                    finalize_exponent = fx.max(fx.Int32(1), fx.min(finalize_exponent, fx.Int32(254)))
                    finalize_hw_scale = (finalize_exponent << fx.Int32(23)).bitcast(fx.Float32)
                    finalize_words = [fx.Int32(0) for finalize__ in range_constexpr(2)]
                    for finalize_word in range_constexpr(2):
                        finalize_q = fx.Int32(llvm.inline_asm(T.i32, [finalize_raw[finalize_word * 2].ir_value(), finalize_hw_scale.ir_value(), fx.Int32(0).ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2', '=v,v,v,0', has_side_effects=True))
                        finalize_words[finalize_word] = fx.Int32(llvm.inline_asm(T.i32, [finalize_raw[finalize_word * 2 + 1].ir_value(), finalize_hw_scale.ir_value(), finalize_q.ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2 op_sel:[0,0,1]', '=v,v,v,0', has_side_effects=True))
                    llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(finalize_activation + finalize_row * 6144 + finalize_head * 128 + finalize_split * 32 + finalize_lane * 8)).ir_value(), fx.Vector.from_elements(finalize_words, fx.Int32).ir_value()], 'global_store_dwordx2 $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                    if finalize_lane == 0:
                        llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(finalize_scales + finalize_head * 64 + finalize_split * 16 + finalize_row)).ir_value(), fx.Int32(finalize_exponent).ir_value()], 'global_store_byte $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                if finalize_block == 0:
                    for finalize_step in range_constexpr(13):
                        finalize_index = finalize_tid + finalize_step * 256
                        if finalize_index < 3264:
                            if finalize_index >= 3072 or finalize_index % 16 >= 8:
                                llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(finalize_scales + finalize_index)).ir_value(), fx.Int32(127).ir_value()], 'global_store_byte $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)

            def output_stage(output_a, output_a_scale, output_weight, output_weight_scale, output_output, epoch, status_slot):
                result = fx.Int32(2)
                output_tid = iteration_tid
                (output_lane, output_wave) = (output_tid % 64, output_tid // 64)
                output_job = iteration_block
                (output_tile, output_peer) = (output_job // 3, output_job % 3)
                if (alive == 1) & (iteration_block < 240):
                    output_reduction = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                    output_demangle = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(output_reduction))

                    def output_uniform(output_v):
                        return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(output_v).ir_value()))

                    def output_descriptor(output_p, output_size):
                        output_addr = fx.Int64(fx.ptrtoint(output_p))
                        return fx.Vector.from_elements([output_uniform(output_addr.to(fx.Int32)), output_uniform(output_addr.shrui(fx.Int64(32)).to(fx.Int32)), fx.Int32(output_size), fx.Int32(159744)], fx.Int32)

                    def output_raw16(output_desc, output_offset, output_aux):
                        return fx.Vector(llvm.call_intrinsic(T.vec(4, T.i32), 'llvm.amdgcn.raw.buffer.load.v4i32', [output_desc.ir_value(), fx.Int32(output_offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(output_aux).ir_value()], [], []))

                    def output_raw4(output_desc, output_offset):
                        return fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.raw.buffer.load.i32', [output_desc.ir_value(), fx.Int32(output_offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value()], [], []))

                    def output_pin4(output_v):
                        return fx.Vector(llvm.inline_asm(T.vec(4, T.i32), [output_v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))

                    def output_pin1(output_v):
                        return fx.Int32(llvm.inline_asm(T.i32, [output_v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))
                    output_ad = output_descriptor(output_a, 49152)
                    output_asd = output_descriptor(output_a_scale, 3264)
                    output_bd = output_descriptor(output_weight, 15728640)
                    output_bsd = output_descriptor(output_weight_scale, 983040)
                    output_alo = [fx.Vector.filled(4, 0, fx.Int32) for output__ in range_constexpr(4)]
                    output_ahi = [fx.Vector.filled(4, 0, fx.Int32) for output__ in range_constexpr(4)]
                    output_asc = [fx.Int32(0) for output__ in range_constexpr(4)]
                    output_bv = [fx.Vector.filled(4, 0, fx.Int32) for output__ in range_constexpr(16)]
                    output_bs = [fx.Int32(0) for output__ in range_constexpr(16)]
                    output_acc = [fx.Vector.filled(4, 0.0, fx.Float32) for output__ in range_constexpr(4)]
                    for output_r in range_constexpr(4):
                        output_kb = output_peer * 16 + output_r * 4 + output_wave
                        for output_atom in range_constexpr(4):
                            output_bv[output_r * 4 + output_atom] = output_raw16(output_bd, ((output_tile * 4 + output_atom) * 48 + output_kb) * 1024 + output_lane * 16, 2)
                            output_bs[output_r * 4 + output_atom] = output_raw4(output_bsd, ((output_tile * 4 + output_atom) * 48 + output_kb) * 64 + (output_lane & fx.Int32(-4)))
                    # Up L2 pre-touch (Delta layers, CTAs 192-239) behind the output prime.
                    if (iteration_block >= 192) & (iteration_block < 224) & (layer % 4 != 3):
                        up_l2_touch(3 + (iteration_block - 192) // 8, 1, 5)
                        up_l2_touch(7 + 2 * ((iteration_block - 192) // 8), 2, 1, 16)
                    if (iteration_block >= 224) & (layer % 4 != 3):
                        up_l2_touch(15 + 7 * ((iteration_block - 224) // 8), 7, 1, 16)
                    # this K-slice's XCD-local ready word (relayed), scalar poll.
                    if output_tid == 0:
                        ol_target = layer + 1
                        ol_ready = fx.Int32(0)
                        ol_polls = fx.Int32(0)
                        while (ol_ready == 0) & (ol_polls < 4194304):
                            ol_ready = (scalar_word(native + CHUNK_WORD + 1408 + (safe_xcc * 3 + output_peer) * 16) >= ol_target).to(fx.Int32)
                            llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                            ol_polls = ol_polls + 1
                        ol_answer = fx.Int32(1)
                        if ol_ready == 0:
                            fail_native()
                            ol_answer = fx.Int32(2)
                        fx.ptr_store(ol_answer, status_ptr + status_slot)
                    gpu.barrier()
                    result = fx.Int32(fx.ptr_load(status_ptr + status_slot))
                    if result == 1:
                        for output_r in range_constexpr(4):
                            output_kb = output_peer * 16 + output_r * 4 + output_wave
                            output_off = output_lane % 16 * 6144 + output_kb * 128 + output_lane // 16 * 16
                            # system-coherent mixer_A loads (no buffer_inv acquire).
                            output_alo[output_r] = output_raw16(output_ad, output_off, 17)
                            output_ahi[output_r] = output_raw16(output_ad, output_off + 64, 17)
                            output_asc[output_r] = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.raw.buffer.load.i32', [output_asd.ir_value(), fx.Int32(output_kb * 64 + (output_lane & fx.Int32(-4))).ir_value(), fx.Int32(0).ir_value(), fx.Int32(17).ir_value()], [], []))
                        for output_k in range_constexpr(4):
                            output_ar = output_k
                            output_br = output_k
                            output_lo = output_pin4(output_alo[output_ar])
                            output_hi = output_pin4(output_ahi[output_ar])
                            output_sa = output_pin1(output_asc[output_ar])
                            output_sa = output_sa.shrui((output_lane & fx.Int32(3)) * 8) & fx.Int32(255)
                            output_av = fx.Vector.from_elements([output_lo[0], output_lo[1], output_lo[2], output_lo[3], output_hi[0], output_hi[1], output_hi[2], output_hi[3]], fx.Int32)
                            for output_atom in range_constexpr(4):
                                output_b = output_pin4(output_bv[output_br * 4 + output_atom])
                                output_sb = output_pin1(output_bs[output_br * 4 + output_atom])
                                output_sb = output_sb.shrui((output_lane & fx.Int32(3)) * 8) & fx.Int32(255)
                                output_bf = fx.Vector.from_elements([output_b[0], output_b[1], output_b[2], output_b[3], fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0)], fx.Int32)
                                output_acc[output_atom] = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [output_av, output_bf, output_acc[output_atom], 0, 4, 0, output_sa, 0, output_sb]))
                        for output_atom in range_constexpr(4):
                            for output_e in range_constexpr(4):
                                fx.ptr_store(output_acc[output_atom][output_e], output_reduction + output_atom * 1024 + output_wave * 256 + output_lane * 4 + output_e)
                        gpu.barrier()
                        if output_wave == 0:
                            for output_atom in range_constexpr(4):
                                output_folded = [fx.Float32(0.0) for output__ in range_constexpr(4)]
                                for output_e in range_constexpr(4):
                                    output_off = output_atom * 1024 + output_lane * 4 + output_e
                                    output_value = fx.ptr_load(output_reduction + output_off)
                                    for output_w in range_constexpr(1, 4):
                                        output_value = output_value + fx.ptr_load(output_reduction + output_off + output_w * 256)
                                    output_folded[output_e] = output_value
                                output_rounded = fx.Vector.from_elements(output_folded, fx.Float32).to(fx.BFloat16)
                                for output_e in range_constexpr(4):
                                    fx.ptr_store(output_rounded[output_e], output_demangle + output_atom * 320 + (output_lane // 16 * 4 + output_e) * 20 + output_lane % 16)
                        gpu.barrier()
                        if output_wave == 0:
                            output_row = output_lane // 4
                            output_col = output_lane % 4 * 4
                            if output_row < 8:
                                for output_atom in range_constexpr(4):
                                    output_packed = fx.Vector(fx.ptr_load(output_demangle + output_atom * 320 + output_row * 20 + output_col, result_type=T.vec(4, T.bf16))).bitcast(fx.Int32)
                                    for output_pair in range_constexpr(2):
                                        output_address = fx.Int64(fx.ptrtoint(output_output + output_row * 5120 + output_tile * 64 + output_atom * 16 + output_col + output_pair * 2))
                                        llvm.InlineAsmOp(None, [output_address.ir_value(), output_packed[output_pair].ir_value()], 'global_atomic_pk_add_bf16 $0, $1, off', 'v,v,~{memory}', has_side_effects=True)
                            # mixer atomics acknowledged -> output chunk counter.
                            llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                            if iteration_tid == 0:
                                system_atomic_add_nw(native + CHUNK_WORD + 128 + output_tile // 10 * 16)
                        gpu.barrier()
                else:
                    result = alive
                return result

            def up_stage(up_a, up_a_scale, up_weight, up_weight_scale, up_output, up_output_scale, epoch, status_slot):
                result = fx.Int32(2)
                up_tid = iteration_tid
                (up_lane, up_wave) = (up_tid % 64, up_tid // 64)
                up_high = (iteration_block >= 144).to(fx.Int32)
                up_unit = iteration_block - 1 - up_high * 7
                up_three = (up_unit < 48).to(fx.Int32)
                up_group = up_three * (up_unit * 3 + up_wave) + (1 - up_three) * (144 + (up_unit - 48) * 2 + up_wave)
                up_active = up_wave < 2 + up_three
                if (alive == 1) & (iteration_block > 0) & ((iteration_block <= 136) | (iteration_block >= 144)):
                    up_shared = arena

                    def up_uniform(up_v):
                        return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(up_v).ir_value()))

                    def up_descriptor(up_p, up_size):
                        up_addr = fx.Int64(fx.ptrtoint(up_p))
                        up_lo = up_uniform(up_addr.to(fx.Int32))
                        up_hi = up_uniform(up_addr.shrui(fx.Int64(32)).to(fx.Int32))
                        return fx.Vector.from_elements([up_lo, up_hi, fx.Int32(up_size), fx.Int32(159744)], fx.Int32)

                    def up_raw16(up_desc, up_offset):
                        return fx.Vector(llvm.inline_asm(T.vec(4, T.i32), [fx.Int32(up_offset).ir_value(), up_desc.ir_value()], 'buffer_load_dwordx4 $0, $1, $2, 0 offen nt', '=&v,v,s,~{memory}', has_side_effects=True))

                    def up_raw_a16(up_desc, up_offset):
                        return fx.Vector(llvm.inline_asm(T.vec(4, T.i32), [fx.Int32(up_offset).ir_value(), up_desc.ir_value()], 'buffer_load_dwordx4 $0, $1, $2, 0 offen', '=&v,v,s,~{memory}', has_side_effects=True))

                    def up_raw4(up_desc, up_offset):
                        return fx.Int32(llvm.inline_asm(T.i32, [fx.Int32(up_offset).ir_value(), up_desc.ir_value()], 'buffer_load_dword $0, $1, $2, 0 offen nt', '=&v,v,s,~{memory}', has_side_effects=True))

                    def up_raw1(up_desc, up_offset):
                        return fx.Int32(llvm.inline_asm(T.i32, [fx.Int32(up_offset).ir_value(), up_desc.ir_value()], 'buffer_load_ubyte $0, $1, $2, 0 offen', '=&v,v,s,~{memory}', has_side_effects=True))

                    def up_dma16(up_desc, up_offset, up_dst):
                        llvm.InlineAsmOp(None, [up_uniform(up_dst).ir_value(), fx.Int32(up_offset).ir_value(), up_desc.ir_value()], 's_mov_b32 m0, $0\ns_nop 0\nbuffer_load_dwordx4 $1, $2, 0 offen nt lds', 's,v,s,~{memory},~{m0}', has_side_effects=True)

                    def up_wait_vm(up_count):
                        llvm.InlineAsmOp(None, [], f's_waitcnt vmcnt({up_count})', '~{memory}', has_side_effects=True)

                    def up_pin4(up_v):
                        return fx.Vector(llvm.inline_asm(T.vec(4, T.i32), [up_v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))

                    def up_pin1(up_v):
                        return fx.Int32(llvm.inline_asm(T.i32, [up_v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))

                    def up_peer8(up_v):
                        return fx.Int32(llvm.inline_asm(T.i32, [up_v.ir_value()], 's_nop 1\nv_mov_b32_dpp $0, $1 row_ror:8 row_mask:0xf bank_mask:0xf', '=&v,v', has_side_effects=True))

                    def up_round_bf16(up_v):
                        up_rounded = up_v.to(fx.BFloat16).bitcast(fx.Int16).to(fx.Int32)
                        up_opaque = fx.Int32(llvm.inline_asm(T.i32, [up_rounded.ir_value()], '', '=v,0', has_side_effects=True))
                        return up_opaque.to(fx.Int16).bitcast(fx.BFloat16).to(fx.Float32)
                    up_ad = up_descriptor(up_a, 40960)
                    up_asd = up_descriptor(up_a_scale, 2752)
                    # every wave streams (k-blocks 4j + wave of all the CTA's atoms). One code
                    # path for both CTA kinds: 2-group CTAs run the third group on zero-size
                    # descriptors (no traffic, zero contributions, never stored).
                    up_bd = up_descriptor(up_weight, 89128960)
                    up_bsd = up_descriptor(up_weight_scale, 5570560)
                    up_bd3 = up_descriptor(up_weight, up_uniform(up_three * 89128960))
                    up_bsd3 = up_descriptor(up_weight_scale, up_uniform(up_three * 5570560))
                    up_ldsbase = up_uniform(fx.ptrtoint(up_shared))
                    up_abase = up_lane % 8 * 5120 + up_lane // 16 * 16 + up_lane % 16 // 8 * 64
                    up_partials = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))

                    # Compiler-tracked scale/activation loads: asm loads whose outputs live across
                    # the post-norm poll loop were copied by the register allocator before their
                    # data returned (group-0 scales of the polling wave came out as garbage).
                    def up_t4(up_desc, up_offset, up_aux):
                        return fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.raw.buffer.load.i32', [up_desc.ir_value(), fx.Int32(up_offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(up_aux).ir_value()], [], []))

                    def up_t16(up_desc, up_offset):
                        return fx.Vector(llvm.call_intrinsic(T.vec(4, T.i32), 'llvm.amdgcn.raw.buffer.load.v4i32', [up_desc.ir_value(), fx.Int32(up_offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value()], [], []))

                    def up_t1(up_desc, up_offset):
                        return fx.Uint8(llvm.call_intrinsic(T.i8, 'llvm.amdgcn.raw.buffer.load.i8', [up_desc.ir_value(), fx.Int32(up_offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value()], [], [])).to(fx.Int32)
                    up_gbase = up_three * (up_unit * 3) + (1 - up_three) * (144 + (up_unit - 48) * 2)
                    up_rbase = up_ldsbase + up_uniform(up_wave) * 39936
                    up_bsc = [[fx.Int32(0) for up__ in range_constexpr(3)] for up__ in range_constexpr(10)]
                    for up_j in range_constexpr(3):
                        up_kb = up_j * 4 + up_wave
                        for up_g in range_constexpr(3):
                            up_bsc[up_j][up_g] = up_t4(up_bsd3 if up_g == 2 else up_bsd, (up_kb * 544 + up_gbase + up_g) * 256 + up_lane * 4, 2)
                        for up_i in range_constexpr(12):
                            up_dma16((up_bd3 if up_i >= 8 else up_bd), (((up_gbase + up_i // 4) * 4 + up_i % 4) * 40 + up_kb) * 1024 + up_lane * 16, up_rbase + up_j * 12288 + up_i * 1024)
                    # steps [3, 4] straight into VGPRs (compiler-tracked loads).
                    up_vb = [[fx.Vector.filled(4, 0, fx.Int32) for up__ in range_constexpr(12)] for up__ in range_constexpr(2)]
                    for up_v in range_constexpr(2):
                        up_kb = (3 + up_v) * 4 + up_wave
                        for up_g in range_constexpr(3):
                            up_bsc[3 + up_v][up_g] = up_t4(up_bsd3 if up_g == 2 else up_bsd, (up_kb * 544 + up_gbase + up_g) * 256 + up_lane * 4, 2)
                        for up_i in range_constexpr(12):
                            up_vb[up_v][up_i] = up_t16(up_bd3 if up_i >= 8 else up_bd, (((up_gbase + up_i // 4) * 4 + up_i % 4) * 40 + up_kb) * 1024 + up_lane * 16)
                    result = up_local_release(status_slot, layer)
                    if result == 1:
                        # system-coherent activation loads instead of an L2 invalidate.
                        up_alo = [fx.Vector(llvm.call_intrinsic(T.vec(4, T.i32), 'llvm.amdgcn.raw.buffer.load.v4i32', [up_ad.ir_value(), fx.Int32(up_abase + (up_j * 4 + up_wave) * 128).ir_value(), fx.Int32(0).ir_value(), fx.Int32(17).ir_value()], [], [])) for up_j in range_constexpr(10)]
                        up_asc = [fx.Uint8(llvm.call_intrinsic(T.i8, 'llvm.amdgcn.raw.buffer.load.i8', [up_asd.ir_value(), fx.Int32((up_j * 4 + up_wave) * 64 + up_lane).ir_value(), fx.Int32(0).ir_value(), fx.Int32(17).ir_value()], [], [])).to(fx.Int32) for up_j in range_constexpr(10)]
                        up_wait_vm(0)
                        # Pin everything loaded before / at the release: no compiler waits on these later.
                        up_vb = [[up_pin4(up_vb[up_v][up_i]) for up_i in range_constexpr(12)] for up_v in range_constexpr(2)]
                        up_alo = [up_pin4(up_alo[up_j]) for up_j in range_constexpr(10)]
                        up_asc = [up_pin1(up_asc[up_j]) for up_j in range_constexpr(10)]
                        up_bsc = [[(up_pin1(up_bsc[up_j][up_g]) if up_j < 5 else up_bsc[up_j][up_g]) for up_g in range_constexpr(3)] for up_j in range_constexpr(10)]
                        up_acc = [fx.Vector.filled(4, 0.0, fx.Float32) for up__ in range_constexpr(12)]
                        # step 0
                        up_low = up_alo[0]
                        up_sa = up_asc[0]
                        up_sbs = [up_bsc[0][up_g] for up_g in range_constexpr(3)]
                        up_av = fx.Vector.from_elements([up_low[0], up_low[1], up_low[2], up_low[3], up_peer8(up_low[0]), up_peer8(up_low[1]), up_peer8(up_low[2]), up_peer8(up_low[3])], fx.Int32)
                        up_bv = [fx.Vector(fx.ptr_load(up_shared + (up_rbase - up_ldsbase) + 0 * 12288 + up_i * 1024 + up_lane * 16, result_type=T.vec(16, T.i8))).bitcast(fx.Int32) for up_i in range_constexpr(12)]
                        rocdl.s_waitcnt(lgkmcnt=0)
                        up_kb = 5 * 4 + up_wave
                        for up_i in range_constexpr(12):
                            up_dma16((up_bd3 if up_i >= 8 else up_bd), (((up_gbase + up_i // 4) * 4 + up_i % 4) * 40 + up_kb) * 1024 + up_lane * 16, up_rbase + 0 * 12288 + up_i * 1024)
                        for up_g in range_constexpr(3):
                            up_bsc[5][up_g] = up_t4(up_bsd3 if up_g == 2 else up_bsd, (up_kb * 544 + up_gbase + up_g) * 256 + up_lane * 4, 2)
                        for up_i in range_constexpr(12):
                            up_acc[up_i] = fx.Vector(llvm.inline_asm(T.vec(4, T.f32), [up_av.ir_value(), up_bv[up_i].ir_value(), up_sa.ir_value(), up_sbs[up_i // 4].ir_value(), up_acc[up_i].ir_value()], f'v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel:[0,{up_i % 2},0] op_sel_hi:[0,{up_i % 4 // 2},0] cbsz:0 blgp:4', '=a,v,v,v,v,0', has_side_effects=True))
                        # step 1
                        up_low = up_alo[1]
                        up_sa = up_asc[1]
                        up_sbs = [up_bsc[1][up_g] for up_g in range_constexpr(3)]
                        up_av = fx.Vector.from_elements([up_low[0], up_low[1], up_low[2], up_low[3], up_peer8(up_low[0]), up_peer8(up_low[1]), up_peer8(up_low[2]), up_peer8(up_low[3])], fx.Int32)
                        up_bv = [fx.Vector(fx.ptr_load(up_shared + (up_rbase - up_ldsbase) + 1 * 12288 + up_i * 1024 + up_lane * 16, result_type=T.vec(16, T.i8))).bitcast(fx.Int32) for up_i in range_constexpr(12)]
                        rocdl.s_waitcnt(lgkmcnt=0)
                        up_kb = 6 * 4 + up_wave
                        for up_i in range_constexpr(12):
                            up_dma16((up_bd3 if up_i >= 8 else up_bd), (((up_gbase + up_i // 4) * 4 + up_i % 4) * 40 + up_kb) * 1024 + up_lane * 16, up_rbase + 1 * 12288 + up_i * 1024)
                        for up_g in range_constexpr(3):
                            up_bsc[6][up_g] = up_t4(up_bsd3 if up_g == 2 else up_bsd, (up_kb * 544 + up_gbase + up_g) * 256 + up_lane * 4, 2)
                        for up_i in range_constexpr(12):
                            up_acc[up_i] = fx.Vector(llvm.inline_asm(T.vec(4, T.f32), [up_av.ir_value(), up_bv[up_i].ir_value(), up_sa.ir_value(), up_sbs[up_i // 4].ir_value(), up_acc[up_i].ir_value()], f'v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel:[0,{up_i % 2},0] op_sel_hi:[0,{up_i % 4 // 2},0] cbsz:0 blgp:4', '=a,v,v,v,v,0', has_side_effects=True))
                        # step 2
                        up_low = up_alo[2]
                        up_sa = up_asc[2]
                        up_sbs = [up_bsc[2][up_g] for up_g in range_constexpr(3)]
                        up_av = fx.Vector.from_elements([up_low[0], up_low[1], up_low[2], up_low[3], up_peer8(up_low[0]), up_peer8(up_low[1]), up_peer8(up_low[2]), up_peer8(up_low[3])], fx.Int32)
                        up_bv = [fx.Vector(fx.ptr_load(up_shared + (up_rbase - up_ldsbase) + 2 * 12288 + up_i * 1024 + up_lane * 16, result_type=T.vec(16, T.i8))).bitcast(fx.Int32) for up_i in range_constexpr(12)]
                        rocdl.s_waitcnt(lgkmcnt=0)
                        up_kb = 7 * 4 + up_wave
                        for up_i in range_constexpr(12):
                            up_dma16((up_bd3 if up_i >= 8 else up_bd), (((up_gbase + up_i // 4) * 4 + up_i % 4) * 40 + up_kb) * 1024 + up_lane * 16, up_rbase + 2 * 12288 + up_i * 1024)
                        for up_g in range_constexpr(3):
                            up_bsc[7][up_g] = up_t4(up_bsd3 if up_g == 2 else up_bsd, (up_kb * 544 + up_gbase + up_g) * 256 + up_lane * 4, 2)
                        for up_i in range_constexpr(12):
                            up_acc[up_i] = fx.Vector(llvm.inline_asm(T.vec(4, T.f32), [up_av.ir_value(), up_bv[up_i].ir_value(), up_sa.ir_value(), up_sbs[up_i // 4].ir_value(), up_acc[up_i].ir_value()], f'v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel:[0,{up_i % 2},0] op_sel_hi:[0,{up_i % 4 // 2},0] cbsz:0 blgp:4', '=a,v,v,v,v,0', has_side_effects=True))
                        # step 3
                        up_low = up_alo[3]
                        up_sa = up_asc[3]
                        up_sbs = [up_bsc[3][up_g] for up_g in range_constexpr(3)]
                        up_av = fx.Vector.from_elements([up_low[0], up_low[1], up_low[2], up_low[3], up_peer8(up_low[0]), up_peer8(up_low[1]), up_peer8(up_low[2]), up_peer8(up_low[3])], fx.Int32)
                        up_bv = up_vb[0]
                        for up_i in range_constexpr(12):
                            up_acc[up_i] = fx.Vector(llvm.inline_asm(T.vec(4, T.f32), [up_av.ir_value(), up_bv[up_i].ir_value(), up_sa.ir_value(), up_sbs[up_i // 4].ir_value(), up_acc[up_i].ir_value()], f'v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel:[0,{up_i % 2},0] op_sel_hi:[0,{up_i % 4 // 2},0] cbsz:0 blgp:4', '=a,v,v,v,v,0', has_side_effects=True))
                        # step 4
                        up_low = up_alo[4]
                        up_sa = up_asc[4]
                        up_sbs = [up_bsc[4][up_g] for up_g in range_constexpr(3)]
                        up_av = fx.Vector.from_elements([up_low[0], up_low[1], up_low[2], up_low[3], up_peer8(up_low[0]), up_peer8(up_low[1]), up_peer8(up_low[2]), up_peer8(up_low[3])], fx.Int32)
                        up_bv = up_vb[1]
                        for up_i in range_constexpr(12):
                            up_acc[up_i] = fx.Vector(llvm.inline_asm(T.vec(4, T.f32), [up_av.ir_value(), up_bv[up_i].ir_value(), up_sa.ir_value(), up_sbs[up_i // 4].ir_value(), up_acc[up_i].ir_value()], f'v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel:[0,{up_i % 2},0] op_sel_hi:[0,{up_i % 4 // 2},0] cbsz:0 blgp:4', '=a,v,v,v,v,0', has_side_effects=True))
                        # step 5
                        up_wait_vm(30)
                        up_low = up_alo[5]
                        up_sa = up_asc[5]
                        up_sbs = [up_bsc[5][up_g] for up_g in range_constexpr(3)]
                        up_av = fx.Vector.from_elements([up_low[0], up_low[1], up_low[2], up_low[3], up_peer8(up_low[0]), up_peer8(up_low[1]), up_peer8(up_low[2]), up_peer8(up_low[3])], fx.Int32)
                        up_bv = [fx.Vector(fx.ptr_load(up_shared + (up_rbase - up_ldsbase) + 0 * 12288 + up_i * 1024 + up_lane * 16, result_type=T.vec(16, T.i8))).bitcast(fx.Int32) for up_i in range_constexpr(12)]
                        rocdl.s_waitcnt(lgkmcnt=0)
                        up_kb = 8 * 4 + up_wave
                        for up_i in range_constexpr(12):
                            up_dma16((up_bd3 if up_i >= 8 else up_bd), (((up_gbase + up_i // 4) * 4 + up_i % 4) * 40 + up_kb) * 1024 + up_lane * 16, up_rbase + 0 * 12288 + up_i * 1024)
                        for up_g in range_constexpr(3):
                            up_bsc[8][up_g] = up_t4(up_bsd3 if up_g == 2 else up_bsd, (up_kb * 544 + up_gbase + up_g) * 256 + up_lane * 4, 2)
                        for up_i in range_constexpr(12):
                            up_acc[up_i] = fx.Vector(llvm.inline_asm(T.vec(4, T.f32), [up_av.ir_value(), up_bv[up_i].ir_value(), up_sa.ir_value(), up_sbs[up_i // 4].ir_value(), up_acc[up_i].ir_value()], f'v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel:[0,{up_i % 2},0] op_sel_hi:[0,{up_i % 4 // 2},0] cbsz:0 blgp:4', '=a,v,v,v,v,0', has_side_effects=True))
                        # step 6
                        up_wait_vm(30)
                        up_low = up_alo[6]
                        up_sa = up_asc[6]
                        up_sbs = [up_bsc[6][up_g] for up_g in range_constexpr(3)]
                        up_av = fx.Vector.from_elements([up_low[0], up_low[1], up_low[2], up_low[3], up_peer8(up_low[0]), up_peer8(up_low[1]), up_peer8(up_low[2]), up_peer8(up_low[3])], fx.Int32)
                        up_bv = [fx.Vector(fx.ptr_load(up_shared + (up_rbase - up_ldsbase) + 1 * 12288 + up_i * 1024 + up_lane * 16, result_type=T.vec(16, T.i8))).bitcast(fx.Int32) for up_i in range_constexpr(12)]
                        rocdl.s_waitcnt(lgkmcnt=0)
                        up_kb = 9 * 4 + up_wave
                        for up_i in range_constexpr(12):
                            up_dma16((up_bd3 if up_i >= 8 else up_bd), (((up_gbase + up_i // 4) * 4 + up_i % 4) * 40 + up_kb) * 1024 + up_lane * 16, up_rbase + 1 * 12288 + up_i * 1024)
                        for up_g in range_constexpr(3):
                            up_bsc[9][up_g] = up_t4(up_bsd3 if up_g == 2 else up_bsd, (up_kb * 544 + up_gbase + up_g) * 256 + up_lane * 4, 2)
                        for up_i in range_constexpr(12):
                            up_acc[up_i] = fx.Vector(llvm.inline_asm(T.vec(4, T.f32), [up_av.ir_value(), up_bv[up_i].ir_value(), up_sa.ir_value(), up_sbs[up_i // 4].ir_value(), up_acc[up_i].ir_value()], f'v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel:[0,{up_i % 2},0] op_sel_hi:[0,{up_i % 4 // 2},0] cbsz:0 blgp:4', '=a,v,v,v,v,0', has_side_effects=True))
                        # step 7
                        up_wait_vm(30)
                        up_low = up_alo[7]
                        up_sa = up_asc[7]
                        up_sbs = [up_bsc[7][up_g] for up_g in range_constexpr(3)]
                        up_av = fx.Vector.from_elements([up_low[0], up_low[1], up_low[2], up_low[3], up_peer8(up_low[0]), up_peer8(up_low[1]), up_peer8(up_low[2]), up_peer8(up_low[3])], fx.Int32)
                        up_bv = [fx.Vector(fx.ptr_load(up_shared + (up_rbase - up_ldsbase) + 2 * 12288 + up_i * 1024 + up_lane * 16, result_type=T.vec(16, T.i8))).bitcast(fx.Int32) for up_i in range_constexpr(12)]
                        rocdl.s_waitcnt(lgkmcnt=0)
                        for up_i in range_constexpr(12):
                            up_acc[up_i] = fx.Vector(llvm.inline_asm(T.vec(4, T.f32), [up_av.ir_value(), up_bv[up_i].ir_value(), up_sa.ir_value(), up_sbs[up_i // 4].ir_value(), up_acc[up_i].ir_value()], f'v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel:[0,{up_i % 2},0] op_sel_hi:[0,{up_i % 4 // 2},0] cbsz:0 blgp:4', '=a,v,v,v,v,0', has_side_effects=True))
                        # step 8
                        up_wait_vm(15)
                        up_low = up_alo[8]
                        up_sa = up_asc[8]
                        up_sbs = [up_bsc[8][up_g] for up_g in range_constexpr(3)]
                        up_av = fx.Vector.from_elements([up_low[0], up_low[1], up_low[2], up_low[3], up_peer8(up_low[0]), up_peer8(up_low[1]), up_peer8(up_low[2]), up_peer8(up_low[3])], fx.Int32)
                        up_bv = [fx.Vector(fx.ptr_load(up_shared + (up_rbase - up_ldsbase) + 0 * 12288 + up_i * 1024 + up_lane * 16, result_type=T.vec(16, T.i8))).bitcast(fx.Int32) for up_i in range_constexpr(12)]
                        rocdl.s_waitcnt(lgkmcnt=0)
                        for up_i in range_constexpr(12):
                            up_acc[up_i] = fx.Vector(llvm.inline_asm(T.vec(4, T.f32), [up_av.ir_value(), up_bv[up_i].ir_value(), up_sa.ir_value(), up_sbs[up_i // 4].ir_value(), up_acc[up_i].ir_value()], f'v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel:[0,{up_i % 2},0] op_sel_hi:[0,{up_i % 4 // 2},0] cbsz:0 blgp:4', '=a,v,v,v,v,0', has_side_effects=True))
                        # step 9
                        up_wait_vm(0)
                        up_low = up_alo[9]
                        up_sa = up_asc[9]
                        up_sbs = [up_bsc[9][up_g] for up_g in range_constexpr(3)]
                        up_av = fx.Vector.from_elements([up_low[0], up_low[1], up_low[2], up_low[3], up_peer8(up_low[0]), up_peer8(up_low[1]), up_peer8(up_low[2]), up_peer8(up_low[3])], fx.Int32)
                        up_bv = [fx.Vector(fx.ptr_load(up_shared + (up_rbase - up_ldsbase) + 1 * 12288 + up_i * 1024 + up_lane * 16, result_type=T.vec(16, T.i8))).bitcast(fx.Int32) for up_i in range_constexpr(12)]
                        rocdl.s_waitcnt(lgkmcnt=0)
                        for up_i in range_constexpr(12):
                            up_acc[up_i] = fx.Vector(llvm.inline_asm(T.vec(4, T.f32), [up_av.ir_value(), up_bv[up_i].ir_value(), up_sa.ir_value(), up_sbs[up_i // 4].ir_value(), up_acc[up_i].ir_value()], f'v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 op_sel:[0,{up_i % 2},0] op_sel_hi:[0,{up_i % 4 // 2},0] cbsz:0 blgp:4', '=a,v,v,v,v,0', has_side_effects=True))
                        up_wait_vm(0)
                        rocdl.s_waitcnt(lgkmcnt=0)
                        # Every wave has left its ring before the partials overwrite arena + 0.
                        gpu.barrier()
                        for up_i in range_constexpr(12):
                            for up_e in range_constexpr(4):
                                fx.ptr_store(up_acc[up_i][up_e], up_partials + ((up_wave * 12 + up_i) * 64 + up_lane) * 4 + up_e)
                    if result == 1:
                        gpu.barrier()
                        # thread t owns output column t % 32 of token row t // 32 in every group.
                        up_c32 = up_tid % 32
                        up_r = up_tid // 32
                        up_atom_t = up_c32 // 8
                        up_cp = up_c32 % 8
                        up_od = up_descriptor(up_output, 139264)
                        up_osd = up_descriptor(up_output_scale, 8896)

                        def up_store_byte(up_desc, up_value_i32, up_offset, up_enable):
                            up_off = up_enable * up_offset + (1 - up_enable) * fx.Int32(0x7FFFFFF0)
                            llvm.call_intrinsic(None, 'llvm.amdgcn.raw.buffer.store.i8', [up_value_i32.to(fx.Int8).ir_value(), up_desc.ir_value(), fx.Int32(up_off).ir_value(), fx.Int32(0).ir_value(), fx.Int32(17).ir_value()], [], [])
                        up_src_lane = up_r // 4 * 16 + up_cp * 2
                        up_src_e = up_r % 4

                        def up_sum_parts(up_g, up_h):
                            up_p = [fx.Float32(fx.ptr_load(up_partials + ((up_w * 12 + up_g * 4 + up_atom_t) * 64 + up_src_lane + up_h) * 4 + up_src_e)) for up_w in range_constexpr(4)]
                            return ((up_p[0] + up_p[1]) + up_p[2]) + up_p[3]
                        up_values = []
                        up_amax = []
                        for up_g in range_constexpr(3):
                            up_gate = up_round_bf16(up_sum_parts(up_g, 0))
                            up_up = up_round_bf16(up_sum_parts(up_g, 1))
                            up_sig = fx.Float32(1.0) / (fx.Float32(1.0) + fmath.exp(-up_gate))
                            up_activated = up_round_bf16(up_gate * up_sig)
                            up_values.append(up_round_bf16(up_activated * up_up))
                            up_amax.append(fmath.absf(up_values[up_g]))
                        for up_shift in (1, 2, 4, 8, 16):
                            up_amax = [fx.max(up_m, up_m.shuffle_xor(up_shift, 64)) for up_m in up_amax]
                        for up_g in range_constexpr(3):
                            up_on = fx.Int32(1) if up_g < 2 else up_three
                            up_grp = up_gbase + up_g
                            up_bits = (up_amax[up_g] * fx.Float32(1.0 / 448.0)).bitcast(fx.Int32)
                            up_exponent = (up_bits.shrui(fx.Int32(23)) & fx.Int32(255)) + ((up_bits & fx.Int32(8388607)) != 0).to(fx.Int32)
                            up_exponent = fx.max(fx.Int32(1), fx.min(up_exponent, fx.Int32(254)))
                            up_scale = (up_exponent << fx.Int32(23)).bitcast(fx.Float32)
                            up_pair = fx.Vector.from_elements([up_values[up_g].to(fx.BFloat16), fx.BFloat16(0.0)], fx.BFloat16).bitcast(fx.Int32)[0]
                            up_packed = fx.Int32(llvm.inline_asm(T.i32, [up_pair.ir_value(), up_scale.ir_value(), fx.Int32(0).ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2', '=v,v,v,0', has_side_effects=False))
                            up_store_byte(up_od, up_packed, up_r * 17408 + up_grp * 32 + up_c32, up_on)
                            up_lead = (up_c32 < 2).to(fx.Int32)
                            up_scale_value = (1 - up_c32) * up_exponent + up_c32 * fx.Int32(127)
                            up_store_byte(up_osd, up_scale_value, up_grp * 16 + up_r + up_c32 * 8, up_lead * up_on)
                        # activated bytes written through on every wave -> K-slice sub-counter.
                        llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                        gpu.barrier()
                        if up_tid == 0:
                            up_slice = (up_unit >= 72).to(fx.Int32) + (up_unit >= 160).to(fx.Int32)
                            system_atomic_add_nw(native + CHUNK_WORD + 256 + (up_slice * 4 + up_unit % 4) * 16)
                else:
                    result = native_postnorm(epoch, status_slot, layer)
                return result

            def down_stage(down_a, down_a_scale, down_weight, down_weight_scale, down_partial, epoch, status_slot):
                result = fx.Int32(2)
                down_tid = iteration_tid
                (down_lane, down_wave) = (down_tid % 64, down_tid // 64)
                # three-group blocks never hold a 12-step (peer-0) Down job (same-XCD swaps).
                down_job = fx.Int32((iteration_block + fx.Int32(iteration_block == 3).to(fx.Int32) * 56 + fx.Int32(iteration_block == 6).to(fx.Int32) * 56 + fx.Int32(iteration_block == 9).to(fx.Int32) * 40 + fx.Int32(iteration_block == 12).to(fx.Int32) * 40 + fx.Int32(iteration_block == 15).to(fx.Int32) * 40 + fx.Int32(iteration_block == 18).to(fx.Int32) * 32 + fx.Int32(iteration_block == 21).to(fx.Int32) * 32 + fx.Int32(iteration_block == 24).to(fx.Int32) * 32 + fx.Int32(iteration_block == 27).to(fx.Int32) * 40 + fx.Int32(iteration_block == 30).to(fx.Int32) * 40 + fx.Int32(iteration_block == 33).to(fx.Int32) * 32 + fx.Int32(iteration_block == 36).to(fx.Int32) * 32 + fx.Int32(iteration_block == 39).to(fx.Int32) * 32 + fx.Int32(iteration_block == 42).to(fx.Int32) * 16 + fx.Int32(iteration_block == 45).to(fx.Int32) * 16 + fx.Int32(iteration_block == 48).to(fx.Int32) * 16 + fx.Int32(iteration_block == 49).to(fx.Int32) * -40 + fx.Int32(iteration_block == 50).to(fx.Int32) * -32 + fx.Int32(iteration_block == 52).to(fx.Int32) * -40 + fx.Int32(iteration_block == 53).to(fx.Int32) * -32 + fx.Int32(iteration_block == 55).to(fx.Int32) * -40 + fx.Int32(iteration_block == 56).to(fx.Int32) * -32 + fx.Int32(iteration_block == 58).to(fx.Int32) * -16 + fx.Int32(iteration_block == 59).to(fx.Int32) * -56 + fx.Int32(iteration_block == 61).to(fx.Int32) * -16 + fx.Int32(iteration_block == 62).to(fx.Int32) * -56 + fx.Int32(iteration_block == 64).to(fx.Int32) * -16 + fx.Int32(iteration_block == 65).to(fx.Int32) * -32 + fx.Int32(iteration_block == 67).to(fx.Int32) * -40 + fx.Int32(iteration_block == 68).to(fx.Int32) * -32 + fx.Int32(iteration_block == 70).to(fx.Int32) * -40 + fx.Int32(iteration_block == 71).to(fx.Int32) * -32))
                (down_tile, down_peer) = (down_job // 3, down_job % 3)
                if (alive == 1) & (iteration_block < 240):
                    down_iterations = fx.Int32(11)
                    if down_peer == 0:
                        down_iterations = fx.Int32(12)
                    down_start = down_peer * 11 + fx.min(down_peer, fx.Int32(1))
                    down_reduction = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))

                    def down_uniform(down_v):
                        return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(down_v).ir_value()))

                    def down_descriptor(down_p, down_size):
                        down_addr = fx.Int64(fx.ptrtoint(down_p))
                        down_lo = down_uniform(down_addr.to(fx.Int32))
                        down_hi = down_uniform(down_addr.shrui(fx.Int64(32)).to(fx.Int32))
                        return fx.Vector.from_elements([down_lo, down_hi, fx.Int32(down_size), fx.Int32(159744)], fx.Int32)

                    def down_raw16(down_desc, down_offset):
                        return fx.Vector(llvm.call_intrinsic(T.vec(4, T.i32), 'llvm.amdgcn.raw.buffer.load.v4i32', [down_desc.ir_value(), fx.Int32(down_offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(2).ir_value()], [], []))

                    def down_raw_a16(down_desc, down_offset):
                        # system-coherent activation loads (no buffer_inv acquire).
                        return fx.Vector(llvm.call_intrinsic(T.vec(4, T.i32), 'llvm.amdgcn.raw.buffer.load.v4i32', [down_desc.ir_value(), fx.Int32(down_offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(17).ir_value()], [], []))

                    def down_c4(down_desc, down_offset):
                        return fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.raw.buffer.load.i32', [down_desc.ir_value(), fx.Int32(down_offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(17).ir_value()], [], []))

                    def down_raw4(down_desc, down_offset):
                        return fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.raw.buffer.load.i32', [down_desc.ir_value(), fx.Int32(down_offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value()], [], []))

                    def down_wait_vm(down_count):
                        llvm.InlineAsmOp(None, [], f's_waitcnt vmcnt({down_count})', '~{memory}', has_side_effects=True)

                    def down_pin4(down_v):
                        return fx.Vector(llvm.inline_asm(T.vec(4, T.i32), [down_v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))

                    def down_pin1(down_v):
                        return fx.Int32(llvm.inline_asm(T.i32, [down_v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))
                    down_ad = down_descriptor(down_a, 139264)
                    down_asd = down_descriptor(down_a_scale, 8896)
                    down_bd = down_descriptor(down_weight, 44564480)
                    down_bsd = down_descriptor(down_weight_scale, 2785280)
                    down_alo = [fx.Vector.filled(4, 0, fx.Int32) for down__ in range_constexpr(12)]
                    down_ahi = [fx.Vector.filled(4, 0, fx.Int32) for down__ in range_constexpr(12)]
                    down_asc = [fx.Int32(0) for down__ in range_constexpr(12)]
                    down_bv = [fx.Vector.filled(4, 0, fx.Int32) for down__ in range_constexpr(36)]
                    down_bs = [fx.Int32(0) for down__ in range_constexpr(36)]
                    down_acc = [fx.Vector.filled(4, 0.0, fx.Float32) for down__ in range_constexpr(4)]
                    # P = 9 weight k-steps primed before the Up -> Down wait.
                    for down_r in range_constexpr(9):
                        down_kb = (down_start + down_r) * 4 + down_wave
                        for down_atom in range_constexpr(4):
                            down_bv[down_r * 4 + down_atom] = down_raw16(down_bd, ((down_tile * 4 + down_atom) * 136 + down_kb) * 1024 + down_lane * 16)
                            down_bs[down_r * 4 + down_atom] = down_raw4(down_bsd, (down_kb * 320 + down_tile * 4 + down_atom) * 64 + (down_lane & fx.Int32(-4)))
                    # this K-slice's XCD-local ready word (relayed), scalar poll.
                    if down_tid == 0:
                        dl_target = layer + 1
                        dl_ready = fx.Int32(0)
                        dl_polls = fx.Int32(0)
                        while (dl_ready == 0) & (dl_polls < 4194304):
                            dl_ready = (scalar_word(native + CHUNK_WORD + 1024 + (safe_xcc * 3 + down_peer) * 16) >= dl_target).to(fx.Int32)
                            llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                            dl_polls = dl_polls + 1
                        dl_answer = fx.Int32(1)
                        if dl_ready == 0:
                            fail_native()
                            dl_answer = fx.Int32(2)
                        fx.ptr_store(dl_answer, status_ptr + status_slot)
                    gpu.barrier()
                    result = fx.Int32(fx.ptr_load(status_ptr + status_slot))
                    if result == 1:
                        # all A activation k-steps right after the wait (was 5, refilled in the loop).
                        for down_r in range_constexpr(12):
                            down_kb = (down_start + down_r) * 4 + down_wave
                            down_off = down_lane % 16 * 17408 + down_kb * 128 + down_lane // 16 * 16
                            down_alo[down_r] = down_raw_a16(down_ad, down_off)
                            down_ahi[down_r] = down_raw_a16(down_ad, down_off + 64)
                            down_asc[down_r] = down_c4(down_asd, down_kb * 64 + (down_lane & fx.Int32(-4)))
                        down_wait_vm(0)
                        for down_k in range_constexpr(12):
                            if down_k < down_iterations:
                                down_r = down_k % 12
                                down_wr = down_k % 9
                                down_lo = down_pin4(down_alo[down_r])
                                down_hi = down_pin4(down_ahi[down_r])
                                down_sa = down_pin1(down_asc[down_r])
                                down_sa = down_sa.shrui((down_lane & fx.Int32(3)) * 8) & fx.Int32(255)
                                down_av = fx.Vector.from_elements([down_lo[0], down_lo[1], down_lo[2], down_lo[3], down_hi[0], down_hi[1], down_hi[2], down_hi[3]], fx.Int32)
                                for down_atom in range_constexpr(4):
                                    down_b = down_pin4(down_bv[down_wr * 4 + down_atom])
                                    down_sb = down_pin1(down_bs[down_wr * 4 + down_atom])
                                    down_sb = down_sb.shrui((down_lane & fx.Int32(3)) * 8) & fx.Int32(255)
                                    down_bf = fx.Vector.from_elements([down_b[0], down_b[1], down_b[2], down_b[3], fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0)], fx.Int32)
                                    down_acc[down_atom] = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [down_av, down_bf, down_acc[down_atom], 0, 4, 0, down_sa, 0, down_sb]))
                                if down_k + 12 < down_iterations:
                                    down_kb = (down_start + down_k + 12) * 4 + down_wave
                                    down_off = down_lane % 16 * 17408 + down_kb * 128 + down_lane // 16 * 16
                                    down_alo[down_r] = down_raw_a16(down_ad, down_off)
                                    down_ahi[down_r] = down_raw_a16(down_ad, down_off + 64)
                                    down_asc[down_r] = down_c4(down_asd, down_kb * 64 + (down_lane & fx.Int32(-4)))
                                if down_k + 9 < down_iterations:
                                    down_kb = (down_start + down_k + 9) * 4 + down_wave
                                    for down_atom in range_constexpr(4):
                                        down_bv[down_wr * 4 + down_atom] = down_raw16(down_bd, ((down_tile * 4 + down_atom) * 136 + down_kb) * 1024 + down_lane * 16)
                                        down_bs[down_wr * 4 + down_atom] = down_raw4(down_bsd, (down_kb * 320 + down_tile * 4 + down_atom) * 64 + (down_lane & fx.Int32(-4)))
                        down_wait_vm(0)
                        for down_atom in range_constexpr(4):
                            for down_e in range_constexpr(4):
                                fx.ptr_store(down_acc[down_atom][down_e], down_reduction + down_wave * 1024 + down_atom * 256 + down_lane * 4 + down_e)
                        gpu.barrier()
                        if down_wave == 0:
                            for down_atom in range_constexpr(4):
                                for down_e in range_constexpr(4):
                                    down_offset = down_atom * 256 + down_lane * 4 + down_e
                                    down_value = fx.ptr_load(down_reduction + down_offset)
                                    for down_w in range_constexpr(1, 4):
                                        down_value = down_value + fx.ptr_load(down_reduction + down_w * 1024 + down_offset)
                                    down_row = down_lane // 16 * 4 + down_e
                                    if down_row < 8:
                                        system_store_u32(down_partial + down_peer * 40960 + down_row * 5120 + down_tile * 64 + down_atom * 16 + down_lane % 16, down_value.bitcast(fx.Int32))
                            # write-through partials complete -> chunk counter.
                            llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                            if iteration_tid == 0:
                                system_atomic_add_nw(native + CHUNK_WORD + down_tile // 10 * 16)
                        gpu.barrier()
                else:
                    result = alive
                return result

            def fold_stage(fold_partial, fold_output):
                fold_i = iteration_block * 256 + iteration_tid
                fold_p0 = fx.ptr_load(fold_partial + fold_i)
                fold_p1 = fx.ptr_load(fold_partial + 40960 + fold_i)
                fold_p2 = fx.ptr_load(fold_partial + 81920 + fold_i)
                fx.ptr_store((fold_p0 + fold_p1 + fold_p2).to(fx.BFloat16), fold_output + fold_i)

            def attention_qgkv_stage(a, a_scale, weight, weight_scale, raw_output, epoch, status_slot):
                result = fx.Int32(2)
                tid = iteration_tid
                (lane, wave) = (tid % 64, tid // 64)
                tile = iteration_block
                if (alive == 1) & (iteration_block < 224):
                    reduction = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                    demangle = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(reduction))

                    def uniform(v):
                        return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))

                    def descriptor(p, size):
                        addr = fx.Int64(fx.ptrtoint(p))
                        return fx.Vector.from_elements([uniform(addr.to(fx.Int32)), uniform(addr.shrui(fx.Int64(32)).to(fx.Int32)), fx.Int32(size), fx.Int32(159744)], fx.Int32)

                    def raw16(desc, offset, aux):
                        return fx.Vector(llvm.call_intrinsic(T.vec(4, T.i32), 'llvm.amdgcn.raw.buffer.load.v4i32', [desc.ir_value(), fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(aux).ir_value()], [], []))

                    def raw4(desc, offset):
                        return fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.raw.buffer.load.i32', [desc.ir_value(), fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value()], [], []))

                    def pin4(v):
                        return fx.Vector(llvm.inline_asm(T.vec(4, T.i32), [v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))

                    def pin1(v):
                        return fx.Int32(llvm.inline_asm(T.i32, [v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))
                    ad = descriptor(a, 40960)
                    asd = descriptor(a_scale, 2752)
                    bd = descriptor(weight, 36700160)
                    bsd = descriptor(weight_scale, 2293760)
                    alo = [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(10)]
                    ahi = [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(10)]
                    asc = [fx.Int32(0) for _ in range_constexpr(10)]
                    bv = [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(24)]
                    bs = [fx.Int32(0) for _ in range_constexpr(24)]
                    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
                    # P = 6 k-steps primed before the fold wait.
                    for r in range_constexpr(6):
                        kb = r * 4 + wave
                        for atom in range_constexpr(4):
                            bv[r * 4 + atom] = raw16(bd, ((tile * 4 + atom) * 40 + kb) * 1024 + lane * 16, 2)
                            bs[r * 4 + atom] = raw4(bsd, (kb * 896 + tile * 4 + atom) * 64 + (lane & fx.Int32(-4)))
                    result = fold_ready_wait(status_slot)
                    if result == 1:
                        # every activation k-block in one round trip.
                        for r in range_constexpr(10):
                            kb = r * 4 + wave
                            off = lane % 16 * 5120 + kb * 128 + lane // 16 * 16
                            alo[r] = raw16(ad, off, 0)
                            ahi[r] = raw16(ad, off + 64, 0)
                            asc[r] = raw4(asd, kb * 64 + (lane & fx.Int32(-4)))
                        for k in range_constexpr(10):
                            ar = k
                            br = k % 6
                            lo = pin4(alo[ar])
                            hi = pin4(ahi[ar])
                            sa = pin1(asc[ar])
                            sa = sa.shrui((lane & fx.Int32(3)) * 8) & fx.Int32(255)
                            av = fx.Vector.from_elements([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]], fx.Int32)
                            for atom in range_constexpr(4):
                                b = pin4(bv[br * 4 + atom])
                                sb = pin1(bs[br * 4 + atom])
                                sb = sb.shrui((lane & fx.Int32(3)) * 8) & fx.Int32(255)
                                bf = fx.Vector.from_elements([b[0], b[1], b[2], b[3], fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0)], fx.Int32)
                                acc[atom] = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [av, bf, acc[atom], 0, 4, 0, sa, 0, sb]))
                            if fx.const_expr(k + 6 < 10):
                                kb = (k + 6) * 4 + wave
                                for atom in range_constexpr(4):
                                    bv[br * 4 + atom] = raw16(bd, ((tile * 4 + atom) * 40 + kb) * 1024 + lane * 16, 2)
                                    bs[br * 4 + atom] = raw4(bsd, (kb * 896 + tile * 4 + atom) * 64 + (lane & fx.Int32(-4)))
                        for atom in range_constexpr(4):
                            for e in range_constexpr(4):
                                fx.ptr_store(acc[atom][e], reduction + atom * 1024 + wave * 256 + lane * 4 + e)
                        gpu.barrier()
                        if wave == 0:
                            for atom in range_constexpr(4):
                                folded = [fx.Float32(0.0) for _ in range_constexpr(4)]
                                for e in range_constexpr(4):
                                    off = atom * 1024 + lane * 4 + e
                                    value = fx.ptr_load(reduction + off)
                                    for w in range_constexpr(1, 4):
                                        value = value + fx.ptr_load(reduction + off + w * 256)
                                    folded[e] = value
                                rounded = fx.Vector.from_elements(folded, fx.Float32).to(fx.BFloat16)
                                for e in range_constexpr(4):
                                    fx.ptr_store(rounded[e], demangle + atom * 320 + (lane // 16 * 4 + e) * 20 + lane % 16)
                        gpu.barrier()
                        if wave == 0:
                            row = lane // 4
                            col = lane % 4 * 4
                            if row < 8:
                                for atom in range_constexpr(4):
                                    values = fx.ptr_load(demangle + atom * 320 + row * 20 + col, result_type=T.vec(4, T.bf16))
                                    # write-through for the point-to-point qk consumers.
                                    llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(raw_output + row * 14336 + tile * 64 + atom * 16 + col)).ir_value(), values.bitcast(fx.Int32).ir_value()], 'global_store_dwordx2 $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                        llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                        gpu.barrier()
                        if tid == 0:
                            qh_c = tile // 8
                            if tile >= 192:
                                qh_c = 24 + (tile - 192) % 16 // 4
                            system_atomic_add_nw(native + CHUNK_WORD + 3200 + qh_c * 16)
                else:
                    result = alive
                return result

            def attention_qk_stage(position, qgatekv, qgamma, kgamma, query, gate, context, ctx_blocks):
                (row, head) = (iteration_block // 28, iteration_block % 28)
                # independent of the QGKV outputs -- issued before the wait.
                d = iteration_tid
                pos = fx.Int32(fx.ptr_load(position + row))
                gamma = fx.Float32(fx.ptr_load(kgamma + d))
                if head < 24:
                    gamma = fx.Float32(fx.ptr_load(qgamma + d))
                page = fx.Int32(fx.ptr_load(ctx_blocks + row))
                theta = fx.Float32(pos) * fmath.powf(fx.Float32(10000000.0), fx.Float32(d % 32) * fx.Float32(-2 / 64))
                (cosine, sine) = (fmath.cos(theta), fmath.sin(theta))
                # this head's QGKV tiles (8 per attention layer), one lane polls.
                if iteration_tid == 0:
                    qw_target = (layer // 4 + 1) * 8
                    qw_have = fx.Int32(0)
                    qw_polls = fx.Int32(0)
                    while (qw_have < qw_target) & (qw_polls < 4194304):
                        qw_have = coherent_u32(native + CHUNK_WORD + 3200 + head * 16)
                        llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                        qw_polls = qw_polls + 1
                    if qw_have < qw_target:
                        fail_native()
                    llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)\nbuffer_inv sc1', '~{memory}', has_side_effects=True)
                gpu.barrier()
                source = fx.Int32(row * 14336 + 12288 + (head - 24) * 256)
                if head < 24:
                    source = fx.Int32(row * 14336 + head * 512)
                    llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(gate + row * 6144 + head * 256 + d)).ir_value(), fx.Vector.from_elements([fx.BFloat16(fx.ptr_load(qgatekv + source + 256 + d)), fx.BFloat16(0.0)], fx.BFloat16).bitcast(fx.Int32)[0].ir_value()], 'global_store_short $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                value = fx.Float32(fx.ptr_load(qgatekv + source + d))
                sq = value * value
                for qk_shift in (1, 2, 4, 8, 16, 32):
                    sq = sq + sq.shuffle_xor(qk_shift, 64)
                qk_red = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                if d % 64 == 0:
                    fx.ptr_store(sq, qk_red + d // 64)
                gpu.barrier()
                total = (fx.Float32(fx.ptr_load(qk_red + 0)) + fx.Float32(fx.ptr_load(qk_red + 1))) + (fx.Float32(fx.ptr_load(qk_red + 2)) + fx.Float32(fx.ptr_load(qk_red + 3)))
                inv = fx.Float32(rocdl.rsq(T.f32, (total * fx.Float32(1 / 256) + fx.Float32(1e-06)).ir_value()))
                norm = value * inv * gamma
                partner = norm.shuffle_xor(32, 64)
                offset = fx.Int32(row * 6144 + head * 256)
                if head >= 24:
                    kvhead = head - 24
                    base = (page * 4 + kvhead) * 16384
                    offset = fx.Int32(base + pos % 32 * 256)
                    v = fx.ptr_load(qgatekv + row * 14336 + 13312 + kvhead * 256 + d)
                    llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(context + base + 8192 + pos % 32 * 256 + d)).ir_value(), fx.Vector.from_elements([fx.BFloat16(v), fx.BFloat16(0.0)], fx.BFloat16).bitcast(fx.Int32)[0].ir_value()], 'global_store_short $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                out0 = norm.to(fx.BFloat16)
                out1 = norm.to(fx.BFloat16)
                if d < 32:
                    first = norm
                    second = partner
                    out0 = (first * cosine - second * sine).to(fx.BFloat16)
                    out1 = (second * cosine + first * sine).to(fx.BFloat16)
                if d < 32 or d >= 64:
                    if head < 24:
                        llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(query + offset + d)).ir_value(), fx.Vector.from_elements([out0, fx.BFloat16(0.0)], fx.BFloat16).bitcast(fx.Int32)[0].ir_value()], 'global_store_short $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                        if d < 32:
                            llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(query + offset + d + 32)).ir_value(), fx.Vector.from_elements([out1, fx.BFloat16(0.0)], fx.BFloat16).bitcast(fx.Int32)[0].ir_value()], 'global_store_short $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                    else:
                        llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(context + offset + d)).ir_value(), fx.Vector.from_elements([out0, fx.BFloat16(0.0)], fx.BFloat16).bitcast(fx.Int32)[0].ir_value()], 'global_store_short $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                        if d < 32:
                            llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(context + offset + d + 32)).ir_value(), fx.Vector.from_elements([out1, fx.BFloat16(0.0)], fx.BFloat16).bitcast(fx.Int32)[0].ir_value()], 'global_store_short $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                # outputs drained -> kv-head counter (query heads 6j..6j+5 and kv head j).
                llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                gpu.barrier()
                if iteration_tid == 0:
                    qp_j = head // 6
                    if head >= 24:
                        qp_j = head - 24
                    system_atomic_add_nw(native + CHUNK_WORD + 3648 + qp_j * 16)

            def attention_partial_pair(query, context, block_map, tasks, dispatch, partial, maximum, denominator, position):
                tid = iteration_tid
                lane = tid % 64
                wave = tid // 64
                half = wave // 2
                llvm.InlineAsmOp(None, [], 's_setreg_imm32_b32 0xdc1, 1;', '~{memory}', has_side_effects=True)
                ktile = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                vtile = ktile + 8192
                past = fx.Int32(fx.ptr_load(position))
                split_tokens = ((past + 8 + 511) // 512) * 32
                splits = (past + 8 + split_tokens - 1) // split_tokens
                rp = iteration_block // 64
                hs = iteration_block % 64
                for task_index in range(hs, splits * 4, 256):
                    kvhead = task_index // splits
                    split = task_index % splits
                    map_offset = fx.Int32(0)
                    start = split * split_tokens
                    stop = fx.min(start + split_tokens - 1, past + rp * 2 + 1)
                    head_base = kvhead * 6
                    column = lane % 16
                    column_row = rp * 2 + column // 6
                    column_head = column % 6
                    column_stop = fx.min(stop, past + column_row)
                    # an all-old first tile's K / V before the qk -> partial wait.
                    first_tile = start // 32 * 32
                    carried_k = fx.Vector.filled(32, 0, fx.BFloat16)
                    carried_v = fx.Vector.filled(32, 0, fx.BFloat16)
                    if (first_tile <= stop) & (first_tile + 32 <= past):
                        page = fx.Int32(fx.ptr_load(block_map + map_offset + first_tile // 32))
                        prime_k = []
                        prime_v = []
                        for chunk in range_constexpr(4):
                            pack = tid + chunk * 256
                            key = pack // 32
                            d = pack % 32 * 8
                            pos = first_tile + key
                            (k, v) = (fx.Vector.filled(8, 0, fx.BFloat16), fx.Vector.filled(8, 0, fx.BFloat16))
                            if pos >= start and pos <= stop:
                                base = (page * 4 + kvhead) * 16384 + pos % 32 * 256 + d
                                k = fx.Vector(fx.ptr_load(context + base, result_type=T.vec(8, T.bf16)))
                                v = fx.Vector(fx.ptr_load(context + base + 8192, result_type=T.vec(8, T.bf16)))
                            for e in range_constexpr(8):
                                prime_k.append(k[e])
                                prime_v.append(v[e])
                        carried_k = fx.Vector.from_elements(prime_k, fx.BFloat16)
                        carried_v = fx.Vector.from_elements(prime_v, fx.BFloat16)
                    # this task's kv head: queries (6 heads x 8 rows) and new K / V (8 rows).
                    if iteration_tid == 0:
                        pw_target = (layer // 4 + 1) * 56
                        pw_have = fx.Int32(0)
                        pw_polls = fx.Int32(0)
                        while (pw_have < pw_target) & (pw_polls < 4194304):
                            pw_have = coherent_u32(native + CHUNK_WORD + 3648 + kvhead * 16)
                            llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                            pw_polls = pw_polls + 1
                        if pw_have < pw_target:
                            fail_native()
                        llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)\nbuffer_inv sc1', '~{memory}', has_side_effects=True)
                    gpu.barrier()
                    qvectors = [fx.Vector.filled(8, 0, fx.BFloat16) for _ in range_constexpr(8)]
                    if column < 12:
                        for kb in range_constexpr(8):
                            qbase = (column_row * 24 + head_base + column_head) * 256 + kb * 32 + lane // 16 * 4
                            qlo = fx.Vector(fx.ptr_load(query + qbase, result_type=T.vec(4, T.bf16)))
                            qhi = fx.Vector(fx.ptr_load(query + qbase + 16, result_type=T.vec(4, T.bf16)))
                            qvectors[kb] = fx.Vector.from_elements([qlo[e] for e in range_constexpr(4)] + [qhi[e] for e in range_constexpr(4)], fx.BFloat16)
                    maxima = fx.Float32(-3.402823466e+38)
                    total = fx.Float32(0)
                    accum = fx.Vector.filled(32, 0, fx.Float32)
                    if (first_tile <= stop) & (first_tile + 32 > past):
                        page = fx.Int32(fx.ptr_load(block_map + map_offset + first_tile // 32))
                        prime_k = []
                        prime_v = []
                        for chunk in range_constexpr(4):
                            pack = tid + chunk * 256
                            key = pack // 32
                            d = pack % 32 * 8
                            pos = first_tile + key
                            (k, v) = (fx.Vector.filled(8, 0, fx.BFloat16), fx.Vector.filled(8, 0, fx.BFloat16))
                            if pos >= start and pos <= stop:
                                base = (page * 4 + kvhead) * 16384 + pos % 32 * 256 + d
                                k = fx.Vector(fx.ptr_load(context + base, result_type=T.vec(8, T.bf16)))
                                v = fx.Vector(fx.ptr_load(context + base + 8192, result_type=T.vec(8, T.bf16)))
                            for e in range_constexpr(8):
                                prime_k.append(k[e])
                                prime_v.append(v[e])
                        carried_k = fx.Vector.from_elements(prime_k, fx.BFloat16)
                        carried_v = fx.Vector.from_elements(prime_v, fx.BFloat16)
                    for tile in range(start // 32 * 32, stop + 1, 32):
                        for chunk in range_constexpr(4):
                            pack = tid + chunk * 256
                            key = pack // 32
                            d = pack % 32 * 8
                            pos = tile + key
                            k = fx.Vector.from_elements([carried_k[chunk * 8 + e] for e in range_constexpr(8)], fx.BFloat16)
                            v = fx.Vector.from_elements([carried_v[chunk * 8 + e] for e in range_constexpr(8)], fx.BFloat16)
                            for pair in range_constexpr(2):
                                logical = key * 64 + d // 4 + pair
                                col = logical % 64
                                physical = logical ^ (key % 16 | (col & 1) << 4)
                                kpart = fx.Vector.from_elements([k[pair * 4 + e] for e in range_constexpr(4)], fx.BFloat16)
                                fx.ptr_store(kpart, ktile + physical * 4)
                            nt_write = d // 16
                            ki_write = key // 16
                            gtr_write = key % 16 // 4
                            k_in_block = key % 4
                            sub_write = (nt_write * 2 + ki_write) * 4 + gtr_write
                            byte_write = sub_write * 128 + (k_in_block * 32 + d % 16 * 2 ^ (nt_write & 3) * 16)
                            fx.ptr_store(v, vtile + byte_write // 2)
                        next_k = fx.Vector.filled(32, 0, fx.BFloat16)
                        next_v = fx.Vector.filled(32, 0, fx.BFloat16)
                        next_tile = tile + 32
                        if next_tile <= stop:
                            next_page = fx.Int32(fx.ptr_load(block_map + map_offset + next_tile // 32))
                            loaded_k = []
                            loaded_v = []
                            for chunk in range_constexpr(4):
                                pack = tid + chunk * 256
                                key = pack // 32
                                d = pack % 32 * 8
                                pos = next_tile + key
                                (k, v) = (fx.Vector.filled(8, 0, fx.BFloat16), fx.Vector.filled(8, 0, fx.BFloat16))
                                if pos >= start and pos <= stop:
                                    base = (next_page * 4 + kvhead) * 16384 + pos % 32 * 256 + d
                                    k = fx.Vector(fx.ptr_load(context + base, result_type=T.vec(8, T.bf16)))
                                    v = fx.Vector(fx.ptr_load(context + base + 8192, result_type=T.vec(8, T.bf16)))
                                for e in range_constexpr(8):
                                    loaded_k.append(k[e])
                                    loaded_v.append(v[e])
                            next_k = fx.Vector.from_elements(loaded_k, fx.BFloat16)
                            next_v = fx.Vector.from_elements(loaded_v, fx.BFloat16)
                        gpu.barrier()
                        correction = fx.Float32(0)
                        scores = fx.Vector.filled(4, 0, fx.Float32)
                        for kb in range_constexpr(8):
                            slot = (half * 16 + lane % 16) * 64 + lane // 16 ^ lane % 32
                            klo = fx.Vector(fx.ptr_load(ktile + (slot ^ kb * 8) * 4, result_type=T.vec(4, T.bf16)))
                            khi = fx.Vector(fx.ptr_load(ktile + (slot ^ kb * 8 + 4) * 4, result_type=T.vec(4, T.bf16)))
                            key = fx.Vector.from_elements([klo[e] for e in range_constexpr(4)] + [khi[e] for e in range_constexpr(4)], fx.BFloat16)
                            scores = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [key, qvectors[kb], scores]))
                        masked = []
                        for e in range_constexpr(4):
                            pos = tile + half * 16 + lane // 16 * 4 + e
                            score = fx.Float32(-3.402823466e+38)
                            if (pos >= start) & (pos <= column_stop):
                                score = scores[e] * fx.Float32(0.0625 * 1.442695041)
                            masked.append(score)
                        tilemax = fx.max(fx.max(masked[0], masked[1]), fx.max(masked[2], masked[3]))
                        tilemax = fx.max(tilemax, tilemax.shuffle_xor(16, 64))
                        tilemax = fx.max(tilemax, tilemax.shuffle_xor(32, 64))
                        newmax = fx.max(maxima, tilemax)
                        correction = fx.Float32(rocdl.exp2(T.f32, (maxima - newmax).ir_value()))
                        probabilities = []
                        for e in range_constexpr(4):
                            pos = tile + half * 16 + lane // 16 * 4 + e
                            probability = fx.Float32(0)
                            if (pos >= start) & (pos <= column_stop):
                                probability = fx.Float32(rocdl.exp2(T.f32, (masked[e] - newmax).ir_value()))
                            probabilities.append(probability)
                        total = total * correction + (probabilities[0] + probabilities[1] + (probabilities[2] + probabilities[3]))
                        maxima = newmax
                        prob = fx.Vector.from_elements([probabilities[e].to(fx.BFloat16) for e in range_constexpr(4)], fx.BFloat16)
                        updated = []
                        for n in range_constexpr(8):
                            nt = wave % 2 * 8 + n
                            sub_read = (nt * 2 + half) * 4 + lane // 16
                            byte_read = sub_read * 128 + (lane % 16 * 8 ^ (nt & 3) * 16)
                            lds_pointer = llvm.inttoptr(ir.Type.parse('!llvm.ptr<3>'), fx.Int32(fx.ptrtoint(vtile) + byte_read).ir_value())
                            value = fx.Vector(llvm.call_intrinsic(T.vec(4, T.f16), 'llvm.amdgcn.ds.read.tr16.b64.v4f16', [lds_pointer], [], []))
                            old = fx.Vector.from_elements([accum[n * 4 + e] * correction for e in range_constexpr(4)], fx.Float32)
                            result = fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(T.vec(4, T.f32), [value.bitcast(fx.Int16), prob.bitcast(fx.Int16), old]))
                            for e in range_constexpr(4):
                                updated.append(result[e])
                        accum = fx.Vector.from_elements(updated, fx.Float32)
                        gpu.barrier()
                        carried_k = next_k
                        carried_v = next_v
                    if wave % 2 == 0:
                        folded = total + total.shuffle_xor(16, 64)
                        folded = folded + folded.shuffle_xor(32, 64)
                        if lane < 16:
                            fx.ptr_store(maxima, fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 49152)) + half * 16 + lane)
                            fx.ptr_store(folded, fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 49152)) + 32 + half * 16 + lane)
                    gpu.barrier()
                    max0 = fx.Float32(fx.ptr_load(fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 49152)) + lane % 16))
                    max1 = fx.Float32(fx.ptr_load(fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 49152)) + 16 + lane % 16))
                    common = fx.max(max0, max1)
                    weight = fx.Float32(rocdl.exp2(T.f32, (maxima - common).ir_value()))
                    if common < fx.Float32(-3.0e38):
                        weight = fx.Float32(0)
                    column_slot = ((column_row * 4 + kvhead) * splits + split) * 6 + column_head
                    if wave >= 2:
                        for n in range_constexpr(8):
                            nt = wave % 2 * 8 + n
                            for e in range_constexpr(4):
                                dim = nt * 16 + lane // 16 * 4 + e
                                fx.ptr_store(accum[n * 4 + e] * weight, fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 32768)) + lane % 16 * 256 + dim)
                    gpu.barrier()
                    if wave < 2:
                        if (wave == 0) & (lane < 12):
                            w0 = fx.Float32(rocdl.exp2(T.f32, (max0 - common).ir_value()))
                            w1 = fx.Float32(rocdl.exp2(T.f32, (max1 - common).ir_value()))
                            if common < fx.Float32(-3.0e38):
                                w0 = fx.Float32(0)
                                w1 = fx.Float32(0)
                            folded = fx.Float32(fx.ptr_load(fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 49152)) + 32 + lane)) * w0 + fx.Float32(fx.ptr_load(fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 49152)) + 48 + lane)) * w1
                            fx.ptr_store(common, maximum + column_slot)
                            fx.ptr_store(folded, denominator + column_slot)
                        if column < 12:
                            for n in range_constexpr(8):
                                nt = wave % 2 * 8 + n
                                for e in range_constexpr(4):
                                    dim = nt * 16 + lane // 16 * 4 + e
                                    peer = fx.Float32(fx.ptr_load(fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 32768)) + lane % 16 * 256 + dim))
                                    value = (accum[n * 4 + e] * weight + peer).to(fx.BFloat16)
                                    fx.ptr_store(value, partial + column_slot * 256 + dim)
                    gpu.barrier()

            def attention_partial_stage(query, context, block_map, tasks, dispatch, partial, maximum, denominator, position):
                tid = iteration_tid
                lane = tid % 64
                wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 64).ir_value()))
                llvm.InlineAsmOp(None, [], 's_setreg_imm32_b32 0xdc1, 1;', '~{memory}', has_side_effects=True)
                # wave w's K / V tile buffer (32 KB) at arena + 32 KB w.
                # wave w's two 16 KB tile buffers at arena + 32 KB w + 16 KB b (K 8 KB, then V 8 KB).
                kv_base = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0)) + wave * 16384
                kv_addr = fx.Int64(fx.ptrtoint(context))
                kv_desc = fx.Vector.from_elements([fx.Int32(rocdl.readfirstlane(T.i32, kv_addr.to(fx.Int32).ir_value())), fx.Int32(rocdl.readfirstlane(T.i32, kv_addr.shrui(fx.Int64(32)).to(fx.Int32).ir_value())), fx.Int32(2147483632), fx.Int32(159744)], fx.Int32)
                past = fx.Int32(fx.ptr_load(position))
                split_tokens = ((past + 8 + 1023) // 1024) * 32
                splits = (past + 8 + split_tokens - 1) // split_tokens
                rq = iteration_block // 128
                hs = iteration_block % 128
                for task_index in range(hs, splits * 4, 256):
                    kvhead = task_index // splits
                    split = task_index % splits
                    map_offset = fx.Int32(0)
                    start = split * split_tokens
                    stop = fx.min(start + split_tokens - 1, past + rq * 4 + 3)
                    head_base = kvhead * 6
                    # lane i holds the page of token block start / 32 + i.
                    task_pages = fx.Int32(fx.ptr_load(block_map + map_offset + start // 32 + fx.min(lane, fx.max(stop // 32 - start // 32, fx.Int32(0)))))
                    task_pages_hi = fx.Int32(fx.ptr_load(block_map + map_offset + start // 32 + fx.min(lane + 64, fx.max(stop // 32 - start // 32, fx.Int32(0)))))
                    if (iteration_tid == 0) & (stop // 32 - start // 32 >= 128):
                        fail_native()
                    # column group g serves columns 16g + lane % 16 < 24 (row 4rq + c // 6, head c % 6).
                    column = lane % 16
                    column_row = []
                    column_head = []
                    column_stop = []
                    # this wave's tiles are first_tile, first_tile + 128, ...
                    first_tile = start // 32 * 32 + wave * 16
                    # the first tile into buffer 0 before the qk -> partial wait (again after it if it holds new tokens).
                    dm_tile = fx.min(first_tile, stop // 16 * 16)
                    dm_pi = dm_tile // 32 - start // 32
                    dm_lo = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.readlane.i32', [task_pages.ir_value(), fx.min(dm_pi, fx.Int32(63)).ir_value()], [], []))
                    dm_hi = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.readlane.i32', [task_pages_hi.ir_value(), fx.max(fx.min(dm_pi - 64, fx.Int32(63)), fx.Int32(0)).ir_value()], [], []))
                    dm_page = dm_lo + (dm_hi - dm_lo) * fx.Int32(dm_pi >= 64).to(fx.Int32)
                    dm_row = (dm_page * 4 + kvhead) * 16384
                    dm_lds = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(arena + 0) + wave * 32768 + (0) * 16384).ir_value()))
                    llvm.InlineAsmOp(None, [], 's_waitcnt lgkmcnt(0)', '~{memory}', has_side_effects=True)
                    for dm_j in range_constexpr(16):
                        if fx.const_expr(dm_j < 8):
                            dm_key = 2 * dm_j + lane // 32
                            dm_off = dm_row + fx.min(dm_tile + dm_key, stop) % 32 * 256 + (lane % 32 ^ dm_key % 16) * 8
                        else:
                            dm_b = (dm_j - 8) * 1024 + lane * 16
                            dm_nt = dm_b // 512
                            dm_w2 = dm_b % 128 ^ (dm_nt & 3) * 16
                            dm_key2 = dm_b // 128 % 4 * 4 + dm_w2 // 32
                            dm_off = dm_row + 8192 + fx.min(dm_tile + dm_key2, stop) % 32 * 256 + dm_nt * 16 + dm_w2 % 32 // 16 * 8
                        llvm.InlineAsmOp(None, [fx.Int32(rocdl.readfirstlane(T.i32, (dm_lds + dm_j * 1024).ir_value())).ir_value(), fx.Int32(dm_off * 2).ir_value(), kv_desc.ir_value()], 's_mov_b32 m0, $0\ns_nop 0\nbuffer_load_dwordx4 $1, $2, 0 offen lds', 's,v,s,~{memory},~{m0}', has_side_effects=True)
                    # this task's kv head: queries (6 heads x 8 rows) and new K / V (8 rows).
                    if iteration_tid == 0:
                        pw_target = (layer // 4 + 1) * 56
                        pw_have = fx.Int32(0)
                        pw_polls = fx.Int32(0)
                        while (pw_have < pw_target) & (pw_polls < 4194304):
                            pw_have = coherent_u32(native + CHUNK_WORD + 3648 + kvhead * 16)
                            llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                            pw_polls = pw_polls + 1
                        if pw_have < pw_target:
                            fail_native()
                        llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)\nbuffer_inv sc1', '~{memory}', has_side_effects=True)
                    gpu.barrier()
                    if (first_tile <= stop) & (first_tile + 16 > past):
                        dm_tile = fx.min(first_tile, stop // 16 * 16)
                        dm_pi = dm_tile // 32 - start // 32
                        dm_lo = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.readlane.i32', [task_pages.ir_value(), fx.min(dm_pi, fx.Int32(63)).ir_value()], [], []))
                        dm_hi = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.readlane.i32', [task_pages_hi.ir_value(), fx.max(fx.min(dm_pi - 64, fx.Int32(63)), fx.Int32(0)).ir_value()], [], []))
                        dm_page = dm_lo + (dm_hi - dm_lo) * fx.Int32(dm_pi >= 64).to(fx.Int32)
                        dm_row = (dm_page * 4 + kvhead) * 16384
                        dm_lds = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(arena + 0) + wave * 32768 + (0) * 16384).ir_value()))
                        llvm.InlineAsmOp(None, [], 's_waitcnt lgkmcnt(0)', '~{memory}', has_side_effects=True)
                        for dm_j in range_constexpr(16):
                            if fx.const_expr(dm_j < 8):
                                dm_key = 2 * dm_j + lane // 32
                                dm_off = dm_row + fx.min(dm_tile + dm_key, stop) % 32 * 256 + (lane % 32 ^ dm_key % 16) * 8
                            else:
                                dm_b = (dm_j - 8) * 1024 + lane * 16
                                dm_nt = dm_b // 512
                                dm_w2 = dm_b % 128 ^ (dm_nt & 3) * 16
                                dm_key2 = dm_b // 128 % 4 * 4 + dm_w2 // 32
                                dm_off = dm_row + 8192 + fx.min(dm_tile + dm_key2, stop) % 32 * 256 + dm_nt * 16 + dm_w2 % 32 // 16 * 8
                            llvm.InlineAsmOp(None, [fx.Int32(rocdl.readfirstlane(T.i32, (dm_lds + dm_j * 1024).ir_value())).ir_value(), fx.Int32(dm_off * 2).ir_value(), kv_desc.ir_value()], 's_mov_b32 m0, $0\ns_nop 0\nbuffer_load_dwordx4 $1, $2, 0 offen lds', 's,v,s,~{memory},~{m0}', has_side_effects=True)
                    qvectors = []
                    for g in range_constexpr(2):
                        gcol = column + g * 16
                        column_row.append(rq * 4 + gcol // 6)
                        column_head.append(gcol % 6)
                        column_stop.append(fx.min(stop, past + column_row[g]))
                        qv = [fx.Vector.filled(8, 0, fx.BFloat16) for _ in range_constexpr(8)]
                        if gcol < 24:
                            for kb in range_constexpr(8):
                                qbase = (column_row[g] * 24 + head_base + column_head[g]) * 256 + kb * 32 + lane // 16 * 8
                                qv[kb] = fx.Vector(fx.ptr_load(query + qbase, result_type=T.vec(8, T.bf16)))
                        qvectors.append(qv)
                    maxima0 = fx.Float32(-3.402823466e+38)
                    maxima1 = fx.Float32(-3.402823466e+38)
                    total0 = fx.Float32(0)
                    total1 = fx.Float32(0)
                    accum0 = fx.Vector.filled(64, 0, fx.Float32)
                    accum1 = fx.Vector.filled(64, 0, fx.Float32)
                    for tile in range(first_tile, stop + 1, 64):
                        cur = (tile - first_tile) // 64 % 2
                        dm_tile = fx.min(tile + 64, stop // 16 * 16)
                        dm_pi = dm_tile // 32 - start // 32
                        dm_lo = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.readlane.i32', [task_pages.ir_value(), fx.min(dm_pi, fx.Int32(63)).ir_value()], [], []))
                        dm_hi = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.readlane.i32', [task_pages_hi.ir_value(), fx.max(fx.min(dm_pi - 64, fx.Int32(63)), fx.Int32(0)).ir_value()], [], []))
                        dm_page = dm_lo + (dm_hi - dm_lo) * fx.Int32(dm_pi >= 64).to(fx.Int32)
                        dm_row = (dm_page * 4 + kvhead) * 16384
                        dm_lds = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(arena + 0) + wave * 32768 + (1 - cur) * 16384).ir_value()))
                        llvm.InlineAsmOp(None, [], 's_waitcnt lgkmcnt(0)', '~{memory}', has_side_effects=True)
                        for dm_j in range_constexpr(16):
                            if fx.const_expr(dm_j < 8):
                                dm_key = 2 * dm_j + lane // 32
                                dm_off = dm_row + fx.min(dm_tile + dm_key, stop) % 32 * 256 + (lane % 32 ^ dm_key % 16) * 8
                            else:
                                dm_b = (dm_j - 8) * 1024 + lane * 16
                                dm_nt = dm_b // 512
                                dm_w2 = dm_b % 128 ^ (dm_nt & 3) * 16
                                dm_key2 = dm_b // 128 % 4 * 4 + dm_w2 // 32
                                dm_off = dm_row + 8192 + fx.min(dm_tile + dm_key2, stop) % 32 * 256 + dm_nt * 16 + dm_w2 % 32 // 16 * 8
                            llvm.InlineAsmOp(None, [fx.Int32(rocdl.readfirstlane(T.i32, (dm_lds + dm_j * 1024).ir_value())).ir_value(), fx.Int32(dm_off * 2).ir_value(), kv_desc.ir_value()], 's_mov_b32 m0, $0\ns_nop 0\nbuffer_load_dwordx4 $1, $2, 0 offen lds', 's,v,s,~{memory},~{m0}', has_side_effects=True)
                        llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(16)', '~{memory}', has_side_effects=True)
                        ktile = kv_base + cur * 8192
                        vtile = ktile + 4096
                        scores = [[fx.Vector.filled(4, 0, fx.Float32) for g in range_constexpr(2)] for h in range_constexpr(1)]
                        for h in range_constexpr(1):
                            for kb in range_constexpr(8):
                                qk_key = fx.Vector(fx.ptr_load(ktile + (lane % 16 * 512 + ((kb * 4 + lane // 16) ^ lane % 16) * 16) // 2, result_type=T.vec(8, T.bf16)))
                                for g in range_constexpr(2):
                                    scores[h][g] = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [qk_key, qvectors[g][kb], scores[h][g]]))
                        maxima_in = [maxima0, maxima1]
                        total_in = [total0, total1]
                        corrections = []
                        probs = []
                        new_maxima = []
                        new_totals = []
                        for g in range_constexpr(2):
                            masked = []
                            for h in range_constexpr(1):
                                for e in range_constexpr(4):
                                    pos = tile + lane // 16 * 4 + e
                                    score = fx.Float32(-3.402823466e+38)
                                    if (pos >= start) & (pos <= column_stop[g]):
                                        score = scores[h][g][e] * fx.Float32(0.0625 * 1.442695041)
                                    masked.append(score)
                            tilemax = fx.max(fx.max(masked[0], masked[1]), fx.max(masked[2], masked[3]))
                            tilemax = fx.max(tilemax, tilemax.shuffle_xor(16, 64))
                            tilemax = fx.max(tilemax, tilemax.shuffle_xor(32, 64))
                            newmax = fx.max(maxima_in[g], tilemax)
                            correction = fx.Float32(rocdl.exp2(T.f32, (maxima_in[g] - newmax).ir_value()))
                            probabilities = []
                            for h in range_constexpr(1):
                                for e in range_constexpr(4):
                                    pos = tile + lane // 16 * 4 + e
                                    probability = fx.Float32(0)
                                    if (pos >= start) & (pos <= column_stop[g]):
                                        probability = fx.Float32(rocdl.exp2(T.f32, (masked[h * 4 + e] - newmax).ir_value()))
                                    probabilities.append(probability)
                            new_totals.append(total_in[g] * correction + (probabilities[0] + probabilities[1] + (probabilities[2] + probabilities[3])))
                            new_maxima.append(newmax)
                            corrections.append(correction)
                            probs.append(fx.Vector.from_elements([probabilities[e].to(fx.BFloat16) for e in range_constexpr(4)], fx.BFloat16))
                        total0 = new_totals[0]
                        total1 = new_totals[1]
                        maxima0 = new_maxima[0]
                        maxima1 = new_maxima[1]
                        accum_in = [accum0, accum1]
                        updated = [[], []]
                        for nt in range_constexpr(16):
                            sub_read = nt * 4 + lane // 16
                            byte_read = sub_read * 128 + (lane % 16 * 8 ^ (nt & 3) * 16)
                            lds_pointer = llvm.inttoptr(ir.Type.parse('!llvm.ptr<3>'), fx.Int32(fx.ptrtoint(vtile) + byte_read).ir_value())
                            value = fx.Vector(llvm.call_intrinsic(T.vec(4, T.f16), 'llvm.amdgcn.ds.read.tr16.b64.v4f16', [lds_pointer], [], []))
                            for g in range_constexpr(2):
                                old = fx.Vector.from_elements([accum_in[g][nt * 4 + e] * corrections[g] for e in range_constexpr(4)], fx.Float32)
                                result = fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(T.vec(4, T.f32), [value.bitcast(fx.Int16), probs[g].bitcast(fx.Int16), old]))
                                for e in range_constexpr(4):
                                    updated[g].append(result[e])
                        accum0 = fx.Vector.from_elements(updated[0], fx.Float32)
                        accum1 = fx.Vector.from_elements(updated[1], fx.Float32)
                        llvm.InlineAsmOp(None, [], '', '~{memory}', has_side_effects=True)
                    llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                    # merge the four waves' states; exchange [wave][g][column (stride 260)][dim] at arena + 0.
                    scratch = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 147456))
                    exchange = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                    maxima_g = [maxima0, maxima1]
                    total_g = [total0, total1]
                    accum_g = [accum0, accum1]
                    for g in range_constexpr(2):
                        folded = total_g[g] + total_g[g].shuffle_xor(16, 64)
                        folded = folded + folded.shuffle_xor(32, 64)
                        if lane < 16:
                            fx.ptr_store(maxima_g[g], scratch + (g * 4 + wave) * 32 + lane)
                            fx.ptr_store(folded, scratch + (g * 4 + wave) * 32 + 16 + lane)
                    gpu.barrier()
                    wmax = [[fx.Float32(fx.ptr_load(scratch + (g * 4 + w) * 32 + column)) for w in range_constexpr(4)] for g in range_constexpr(2)]
                    common = [fx.max(fx.max(wmax[g][0], wmax[g][1]), fx.max(wmax[g][2], wmax[g][3])) for g in range_constexpr(2)]
                    for g in range_constexpr(2):
                        w_own = fx.Float32(rocdl.exp2(T.f32, (maxima_g[g] - common[g]).ir_value()))
                        if common[g] < fx.Float32(-3.0e38):
                            w_own = fx.Float32(0)
                        for nt in range_constexpr(16):
                            scaled = fx.Vector.from_elements([accum_g[g][nt * 4 + e] * w_own for e in range_constexpr(4)], fx.Float32)
                            fx.ptr_store(scaled, exchange + (wave * 2 + g) * 4160 + column * 260 + nt * 16 + lane // 16 * 4)
                    gpu.barrier()
                    column_slot = [((column_row[g] * 4 + kvhead) * splits + split) * 6 + column_head[g] for g in range_constexpr(2)]
                    for g in range_constexpr(2):
                        gcol = column + g * 16
                        if gcol < 24:
                            if wave == 0:
                                folded = fx.Float32(0)
                                for w in range_constexpr(4):
                                    ww = fx.Float32(rocdl.exp2(T.f32, (wmax[g][w] - common[g]).ir_value()))
                                    if common[g] < fx.Float32(-3.0e38):
                                        ww = fx.Float32(0)
                                    folded = folded + fx.Float32(fx.ptr_load(scratch + (g * 4 + w) * 32 + 16 + column)) * ww
                                fx.ptr_store(common[g], maximum + column_slot[g])
                                fx.ptr_store(folded, denominator + column_slot[g])
                            for part in range_constexpr(2):
                                dim0 = wave * 64 + lane // 16 * 16 + part * 8
                                sums = [fx.Float32(0) for _ in range_constexpr(8)]
                                for w in range_constexpr(4):
                                    for q4 in range_constexpr(2):
                                        piece = fx.Vector(fx.ptr_load(exchange + (w * 2 + g) * 4160 + column * 260 + dim0 + q4 * 4, result_type=T.vec(4, T.f32)))
                                        for e in range_constexpr(4):
                                            sums[q4 * 4 + e] = sums[q4 * 4 + e] + piece[e]
                                fx.ptr_store(fx.Vector.from_elements([sums[e].to(fx.BFloat16) for e in range_constexpr(8)], fx.BFloat16), partial + column_slot[g] * 256 + dim0)
                    gpu.barrier()

            def attention_partial_oct_stage(query, context, block_map, tasks, dispatch, partial, maximum, denominator, position):
                tid = iteration_tid
                lane = tid % 64
                wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 64).ir_value()))
                llvm.InlineAsmOp(None, [], 's_setreg_imm32_b32 0xdc1, 1;', '~{memory}', has_side_effects=True)
                # wave w's K / V tile buffer (32 KB) at arena + 32 KB w.
                # wave w's two 16 KB tile buffers at arena + 32 KB w + 16 KB b (K 8 KB, then V 8 KB).
                kv_base = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0)) + wave * 16384
                kv_addr = fx.Int64(fx.ptrtoint(context))
                kv_desc = fx.Vector.from_elements([fx.Int32(rocdl.readfirstlane(T.i32, kv_addr.to(fx.Int32).ir_value())), fx.Int32(rocdl.readfirstlane(T.i32, kv_addr.shrui(fx.Int64(32)).to(fx.Int32).ir_value())), fx.Int32(2147483632), fx.Int32(159744)], fx.Int32)
                past = fx.Int32(fx.ptr_load(position))
                split_tokens = ((past + 8 + 2047) // 2048) * 32  # 532 row-oct: up to 64 splits
                splits = (past + 8 + split_tokens - 1) // split_tokens
                rq = fx.Int32(0)
                hs = iteration_block
                for task_index in range(hs, splits * 4, 256):
                    kvhead = task_index // splits
                    split = task_index % splits
                    map_offset = fx.Int32(0)
                    start = split * split_tokens
                    stop = fx.min(start + split_tokens - 1, past + 7)
                    head_base = kvhead * 6
                    # lane i holds the page of token block start / 32 + i.
                    task_pages = fx.Int32(fx.ptr_load(block_map + map_offset + start // 32 + fx.min(lane, fx.max(stop // 32 - start // 32, fx.Int32(0)))))
                    task_pages_hi = fx.Int32(fx.ptr_load(block_map + map_offset + start // 32 + fx.min(lane + 64, fx.max(stop // 32 - start // 32, fx.Int32(0)))))
                    if (iteration_tid == 0) & (stop // 32 - start // 32 >= 128):
                        fail_native()
                    # column group g serves columns 16g + lane % 16 < 24 (row 4rq + c // 6, head c % 6).
                    column = lane % 16
                    column_row = []
                    column_head = []
                    column_stop = []
                    # this wave's tiles are first_tile, first_tile + 128, ...
                    first_tile = start // 32 * 32 + wave * 16
                    # the first tile into buffer 0 before the qk -> partial wait (again after it if it holds new tokens).
                    dm_tile = fx.min(first_tile, stop // 16 * 16)
                    dm_pi = dm_tile // 32 - start // 32
                    dm_lo = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.readlane.i32', [task_pages.ir_value(), fx.min(dm_pi, fx.Int32(63)).ir_value()], [], []))
                    dm_hi = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.readlane.i32', [task_pages_hi.ir_value(), fx.max(fx.min(dm_pi - 64, fx.Int32(63)), fx.Int32(0)).ir_value()], [], []))
                    dm_page = dm_lo + (dm_hi - dm_lo) * fx.Int32(dm_pi >= 64).to(fx.Int32)
                    dm_row = (dm_page * 4 + kvhead) * 16384
                    dm_lds = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(arena + 0) + wave * 32768 + (0) * 16384).ir_value()))
                    llvm.InlineAsmOp(None, [], 's_waitcnt lgkmcnt(0)', '~{memory}', has_side_effects=True)
                    for dm_j in range_constexpr(16):
                        if fx.const_expr(dm_j < 8):
                            dm_key = 2 * dm_j + lane // 32
                            dm_off = dm_row + fx.min(dm_tile + dm_key, stop) % 32 * 256 + (lane % 32 ^ dm_key % 16) * 8
                        else:
                            dm_b = (dm_j - 8) * 1024 + lane * 16
                            dm_nt = dm_b // 512
                            dm_w2 = dm_b % 128 ^ (dm_nt & 3) * 16
                            dm_key2 = dm_b // 128 % 4 * 4 + dm_w2 // 32
                            dm_off = dm_row + 8192 + fx.min(dm_tile + dm_key2, stop) % 32 * 256 + dm_nt * 16 + dm_w2 % 32 // 16 * 8
                        llvm.InlineAsmOp(None, [fx.Int32(rocdl.readfirstlane(T.i32, (dm_lds + dm_j * 1024).ir_value())).ir_value(), fx.Int32(dm_off * 2).ir_value(), kv_desc.ir_value()], 's_mov_b32 m0, $0\ns_nop 0\nbuffer_load_dwordx4 $1, $2, 0 offen lds', 's,v,s,~{memory},~{m0}', has_side_effects=True)
                    # this task's kv head: queries (6 heads x 8 rows) and new K / V (8 rows).
                    if iteration_tid == 0:
                        pw_target = (layer // 4 + 1) * 56
                        pw_have = fx.Int32(0)
                        pw_polls = fx.Int32(0)
                        while (pw_have < pw_target) & (pw_polls < 4194304):
                            pw_have = coherent_u32(native + CHUNK_WORD + 3648 + kvhead * 16)
                            llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                            pw_polls = pw_polls + 1
                        if pw_have < pw_target:
                            fail_native()
                        llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)\nbuffer_inv sc1', '~{memory}', has_side_effects=True)
                    gpu.barrier()
                    if (first_tile <= stop) & (first_tile + 16 > past):
                        dm_tile = fx.min(first_tile, stop // 16 * 16)
                        dm_pi = dm_tile // 32 - start // 32
                        dm_lo = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.readlane.i32', [task_pages.ir_value(), fx.min(dm_pi, fx.Int32(63)).ir_value()], [], []))
                        dm_hi = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.readlane.i32', [task_pages_hi.ir_value(), fx.max(fx.min(dm_pi - 64, fx.Int32(63)), fx.Int32(0)).ir_value()], [], []))
                        dm_page = dm_lo + (dm_hi - dm_lo) * fx.Int32(dm_pi >= 64).to(fx.Int32)
                        dm_row = (dm_page * 4 + kvhead) * 16384
                        dm_lds = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(arena + 0) + wave * 32768 + (0) * 16384).ir_value()))
                        llvm.InlineAsmOp(None, [], 's_waitcnt lgkmcnt(0)', '~{memory}', has_side_effects=True)
                        for dm_j in range_constexpr(16):
                            if fx.const_expr(dm_j < 8):
                                dm_key = 2 * dm_j + lane // 32
                                dm_off = dm_row + fx.min(dm_tile + dm_key, stop) % 32 * 256 + (lane % 32 ^ dm_key % 16) * 8
                            else:
                                dm_b = (dm_j - 8) * 1024 + lane * 16
                                dm_nt = dm_b // 512
                                dm_w2 = dm_b % 128 ^ (dm_nt & 3) * 16
                                dm_key2 = dm_b // 128 % 4 * 4 + dm_w2 // 32
                                dm_off = dm_row + 8192 + fx.min(dm_tile + dm_key2, stop) % 32 * 256 + dm_nt * 16 + dm_w2 % 32 // 16 * 8
                            llvm.InlineAsmOp(None, [fx.Int32(rocdl.readfirstlane(T.i32, (dm_lds + dm_j * 1024).ir_value())).ir_value(), fx.Int32(dm_off * 2).ir_value(), kv_desc.ir_value()], 's_mov_b32 m0, $0\ns_nop 0\nbuffer_load_dwordx4 $1, $2, 0 offen lds', 's,v,s,~{memory},~{m0}', has_side_effects=True)
                    qvectors = []
                    for g in range_constexpr(3):
                        gcol = column + g * 16
                        column_row.append(rq * 4 + gcol // 6)
                        column_head.append(gcol % 6)
                        column_stop.append(fx.min(stop, past + column_row[g]))
                        qv = [fx.Vector.filled(8, 0, fx.BFloat16) for _ in range_constexpr(8)]
                        if gcol < 48:
                            for kb in range_constexpr(8):
                                qbase = (column_row[g] * 24 + head_base + column_head[g]) * 256 + kb * 32 + lane // 16 * 8
                                qv[kb] = fx.Vector(fx.ptr_load(query + qbase, result_type=T.vec(8, T.bf16)))
                        qvectors.append(qv)
                    maxima0 = fx.Float32(-3.402823466e+38)
                    maxima1 = fx.Float32(-3.402823466e+38)
                    maxima2 = fx.Float32(-3.402823466e+38)
                    total0 = fx.Float32(0)
                    total1 = fx.Float32(0)
                    total2 = fx.Float32(0)
                    accum0 = fx.Vector.filled(64, 0, fx.Float32)
                    accum1 = fx.Vector.filled(64, 0, fx.Float32)
                    accum2 = fx.Vector.filled(64, 0, fx.Float32)
                    for tile in range(first_tile, stop + 1, 64):
                        cur = (tile - first_tile) // 64 % 2
                        dm_tile = fx.min(tile + 64, stop // 16 * 16)
                        dm_pi = dm_tile // 32 - start // 32
                        dm_lo = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.readlane.i32', [task_pages.ir_value(), fx.min(dm_pi, fx.Int32(63)).ir_value()], [], []))
                        dm_hi = fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.readlane.i32', [task_pages_hi.ir_value(), fx.max(fx.min(dm_pi - 64, fx.Int32(63)), fx.Int32(0)).ir_value()], [], []))
                        dm_page = dm_lo + (dm_hi - dm_lo) * fx.Int32(dm_pi >= 64).to(fx.Int32)
                        dm_row = (dm_page * 4 + kvhead) * 16384
                        dm_lds = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(arena + 0) + wave * 32768 + (1 - cur) * 16384).ir_value()))
                        llvm.InlineAsmOp(None, [], 's_waitcnt lgkmcnt(0)', '~{memory}', has_side_effects=True)
                        for dm_j in range_constexpr(16):
                            if fx.const_expr(dm_j < 8):
                                dm_key = 2 * dm_j + lane // 32
                                dm_off = dm_row + fx.min(dm_tile + dm_key, stop) % 32 * 256 + (lane % 32 ^ dm_key % 16) * 8
                            else:
                                dm_b = (dm_j - 8) * 1024 + lane * 16
                                dm_nt = dm_b // 512
                                dm_w2 = dm_b % 128 ^ (dm_nt & 3) * 16
                                dm_key2 = dm_b // 128 % 4 * 4 + dm_w2 // 32
                                dm_off = dm_row + 8192 + fx.min(dm_tile + dm_key2, stop) % 32 * 256 + dm_nt * 16 + dm_w2 % 32 // 16 * 8
                            llvm.InlineAsmOp(None, [fx.Int32(rocdl.readfirstlane(T.i32, (dm_lds + dm_j * 1024).ir_value())).ir_value(), fx.Int32(dm_off * 2).ir_value(), kv_desc.ir_value()], 's_mov_b32 m0, $0\ns_nop 0\nbuffer_load_dwordx4 $1, $2, 0 offen lds', 's,v,s,~{memory},~{m0}', has_side_effects=True)
                        llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(16)', '~{memory}', has_side_effects=True)
                        ktile = kv_base + cur * 8192
                        vtile = ktile + 4096
                        scores = [[fx.Vector.filled(4, 0, fx.Float32) for g in range_constexpr(3)] for h in range_constexpr(1)]
                        for h in range_constexpr(1):
                            for kb in range_constexpr(8):
                                qk_key = fx.Vector(fx.ptr_load(ktile + (lane % 16 * 512 + ((kb * 4 + lane // 16) ^ lane % 16) * 16) // 2, result_type=T.vec(8, T.bf16)))
                                for g in range_constexpr(3):
                                    scores[h][g] = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [qk_key, qvectors[g][kb], scores[h][g]]))
                        maxima_in = [maxima0, maxima1, maxima2]
                        total_in = [total0, total1, total2]
                        corrections = []
                        probs = []
                        new_maxima = []
                        new_totals = []
                        for g in range_constexpr(3):
                            masked = []
                            for h in range_constexpr(1):
                                for e in range_constexpr(4):
                                    pos = tile + lane // 16 * 4 + e
                                    score = fx.Float32(-3.402823466e+38)
                                    if (pos >= start) & (pos <= column_stop[g]):
                                        score = scores[h][g][e] * fx.Float32(0.0625 * 1.442695041)
                                    masked.append(score)
                            tilemax = fx.max(fx.max(masked[0], masked[1]), fx.max(masked[2], masked[3]))
                            tilemax = fx.max(tilemax, tilemax.shuffle_xor(16, 64))
                            tilemax = fx.max(tilemax, tilemax.shuffle_xor(32, 64))
                            newmax = fx.max(maxima_in[g], tilemax)
                            correction = fx.Float32(rocdl.exp2(T.f32, (maxima_in[g] - newmax).ir_value()))
                            probabilities = []
                            for h in range_constexpr(1):
                                for e in range_constexpr(4):
                                    pos = tile + lane // 16 * 4 + e
                                    probability = fx.Float32(0)
                                    if (pos >= start) & (pos <= column_stop[g]):
                                        probability = fx.Float32(rocdl.exp2(T.f32, (masked[h * 4 + e] - newmax).ir_value()))
                                    probabilities.append(probability)
                            new_totals.append(total_in[g] * correction + (probabilities[0] + probabilities[1] + (probabilities[2] + probabilities[3])))
                            new_maxima.append(newmax)
                            corrections.append(correction)
                            probs.append(fx.Vector.from_elements([probabilities[e].to(fx.BFloat16) for e in range_constexpr(4)], fx.BFloat16))
                        total0 = new_totals[0]
                        total1 = new_totals[1]
                        total2 = new_totals[2]
                        maxima0 = new_maxima[0]
                        maxima1 = new_maxima[1]
                        maxima2 = new_maxima[2]
                        accum_in = [accum0, accum1, accum2]
                        updated = [[], [], []]
                        for nt in range_constexpr(16):
                            sub_read = nt * 4 + lane // 16
                            byte_read = sub_read * 128 + (lane % 16 * 8 ^ (nt & 3) * 16)
                            lds_pointer = llvm.inttoptr(ir.Type.parse('!llvm.ptr<3>'), fx.Int32(fx.ptrtoint(vtile) + byte_read).ir_value())
                            value = fx.Vector(llvm.call_intrinsic(T.vec(4, T.f16), 'llvm.amdgcn.ds.read.tr16.b64.v4f16', [lds_pointer], [], []))
                            for g in range_constexpr(3):
                                old = fx.Vector.from_elements([accum_in[g][nt * 4 + e] * corrections[g] for e in range_constexpr(4)], fx.Float32)
                                result = fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(T.vec(4, T.f32), [value.bitcast(fx.Int16), probs[g].bitcast(fx.Int16), old]))
                                for e in range_constexpr(4):
                                    updated[g].append(result[e])
                        accum0 = fx.Vector.from_elements(updated[0], fx.Float32)
                        accum1 = fx.Vector.from_elements(updated[1], fx.Float32)
                        accum2 = fx.Vector.from_elements(updated[2], fx.Float32)
                        llvm.InlineAsmOp(None, [], '', '~{memory}', has_side_effects=True)
                    llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                    # merge the four waves' states for three column groups (48 columns, all used); the
                    # exchange [slot][column (stride 260)][dim] at arena + 0 is reused in two rounds.
                    scratch = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 147456))
                    exchange = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                    maxima_g = [maxima0, maxima1, maxima2]
                    total_g = [total0, total1, total2]
                    accum_g = [accum0, accum1, accum2]
                    for g in range_constexpr(3):
                        folded = total_g[g] + total_g[g].shuffle_xor(16, 64)
                        folded = folded + folded.shuffle_xor(32, 64)
                        if lane < 16:
                            fx.ptr_store(maxima_g[g], scratch + (g * 4 + wave) * 32 + lane)
                            fx.ptr_store(folded, scratch + (g * 4 + wave) * 32 + 16 + lane)
                    gpu.barrier()
                    wmax = [[fx.Float32(fx.ptr_load(scratch + (g * 4 + w) * 32 + column)) for w in range_constexpr(4)] for g in range_constexpr(3)]
                    common = [fx.max(fx.max(wmax[g][0], wmax[g][1]), fx.max(wmax[g][2], wmax[g][3])) for g in range_constexpr(3)]
                    column_slot = [((column_row[g] * 4 + kvhead) * splits + split) * 6 + column_head[g] for g in range_constexpr(3)]
                    w_own0 = fx.Float32(rocdl.exp2(T.f32, (maxima_g[0] - common[0]).ir_value()))
                    if common[0] < fx.Float32(-3.0e38):
                        w_own0 = fx.Float32(0)
                    for nt in range_constexpr(16):
                        scaled = fx.Vector.from_elements([accum_g[0][nt * 4 + e] * w_own0 for e in range_constexpr(4)], fx.Float32)
                        fx.ptr_store(scaled, exchange + (wave * 2 + 0) * 4160 + column * 260 + nt * 16 + lane // 16 * 4)
                    w_own1 = fx.Float32(rocdl.exp2(T.f32, (maxima_g[1] - common[1]).ir_value()))
                    if common[1] < fx.Float32(-3.0e38):
                        w_own1 = fx.Float32(0)
                    for nt in range_constexpr(16):
                        scaled = fx.Vector.from_elements([accum_g[1][nt * 4 + e] * w_own1 for e in range_constexpr(4)], fx.Float32)
                        fx.ptr_store(scaled, exchange + (wave * 2 + 1) * 4160 + column * 260 + nt * 16 + lane // 16 * 4)
                    gpu.barrier()
                    if wave == 0:
                        folded0 = fx.Float32(0)
                        for w in range_constexpr(4):
                            ww = fx.Float32(rocdl.exp2(T.f32, (wmax[0][w] - common[0]).ir_value()))
                            if common[0] < fx.Float32(-3.0e38):
                                ww = fx.Float32(0)
                            folded0 = folded0 + fx.Float32(fx.ptr_load(scratch + (0 * 4 + w) * 32 + 16 + column)) * ww
                        fx.ptr_store(common[0], maximum + column_slot[0])
                        fx.ptr_store(folded0, denominator + column_slot[0])
                    for part in range_constexpr(2):
                        dim0 = wave * 64 + lane // 16 * 16 + part * 8
                        sums = [fx.Float32(0) for _ in range_constexpr(8)]
                        for w in range_constexpr(4):
                            for q4 in range_constexpr(2):
                                piece = fx.Vector(fx.ptr_load(exchange + (w * 2 + 0) * 4160 + column * 260 + dim0 + q4 * 4, result_type=T.vec(4, T.f32)))
                                for e in range_constexpr(4):
                                    sums[q4 * 4 + e] = sums[q4 * 4 + e] + piece[e]
                        fx.ptr_store(fx.Vector.from_elements([sums[e].to(fx.BFloat16) for e in range_constexpr(8)], fx.BFloat16), partial + column_slot[0] * 256 + dim0)
                    if wave == 0:
                        folded1 = fx.Float32(0)
                        for w in range_constexpr(4):
                            ww = fx.Float32(rocdl.exp2(T.f32, (wmax[1][w] - common[1]).ir_value()))
                            if common[1] < fx.Float32(-3.0e38):
                                ww = fx.Float32(0)
                            folded1 = folded1 + fx.Float32(fx.ptr_load(scratch + (1 * 4 + w) * 32 + 16 + column)) * ww
                        fx.ptr_store(common[1], maximum + column_slot[1])
                        fx.ptr_store(folded1, denominator + column_slot[1])
                    for part in range_constexpr(2):
                        dim0 = wave * 64 + lane // 16 * 16 + part * 8
                        sums = [fx.Float32(0) for _ in range_constexpr(8)]
                        for w in range_constexpr(4):
                            for q4 in range_constexpr(2):
                                piece = fx.Vector(fx.ptr_load(exchange + (w * 2 + 1) * 4160 + column * 260 + dim0 + q4 * 4, result_type=T.vec(4, T.f32)))
                                for e in range_constexpr(4):
                                    sums[q4 * 4 + e] = sums[q4 * 4 + e] + piece[e]
                        fx.ptr_store(fx.Vector.from_elements([sums[e].to(fx.BFloat16) for e in range_constexpr(8)], fx.BFloat16), partial + column_slot[1] * 256 + dim0)
                    gpu.barrier()
                    w_own2 = fx.Float32(rocdl.exp2(T.f32, (maxima_g[2] - common[2]).ir_value()))
                    if common[2] < fx.Float32(-3.0e38):
                        w_own2 = fx.Float32(0)
                    for nt in range_constexpr(16):
                        scaled = fx.Vector.from_elements([accum_g[2][nt * 4 + e] * w_own2 for e in range_constexpr(4)], fx.Float32)
                        fx.ptr_store(scaled, exchange + (wave * 2 + 0) * 4160 + column * 260 + nt * 16 + lane // 16 * 4)
                    gpu.barrier()
                    if wave == 0:
                        folded2 = fx.Float32(0)
                        for w in range_constexpr(4):
                            ww = fx.Float32(rocdl.exp2(T.f32, (wmax[2][w] - common[2]).ir_value()))
                            if common[2] < fx.Float32(-3.0e38):
                                ww = fx.Float32(0)
                            folded2 = folded2 + fx.Float32(fx.ptr_load(scratch + (2 * 4 + w) * 32 + 16 + column)) * ww
                        fx.ptr_store(common[2], maximum + column_slot[2])
                        fx.ptr_store(folded2, denominator + column_slot[2])
                    for part in range_constexpr(2):
                        dim0 = wave * 64 + lane // 16 * 16 + part * 8
                        sums = [fx.Float32(0) for _ in range_constexpr(8)]
                        for w in range_constexpr(4):
                            for q4 in range_constexpr(2):
                                piece = fx.Vector(fx.ptr_load(exchange + (w * 2 + 0) * 4160 + column * 260 + dim0 + q4 * 4, result_type=T.vec(4, T.f32)))
                                for e in range_constexpr(4):
                                    sums[q4 * 4 + e] = sums[q4 * 4 + e] + piece[e]
                        fx.ptr_store(fx.Vector.from_elements([sums[e].to(fx.BFloat16) for e in range_constexpr(8)], fx.BFloat16), partial + column_slot[2] * 256 + dim0)
                    gpu.barrier()

            def attention_reduce_stage(gate, tasks, dispatch, partial, maximum, denominator, gated, out, scales, position):
                tid = iteration_tid
                past = fx.Int32(fx.ptr_load(position))
                split_tokens = ((past + 8 + 511) // 512) * 32
                # the reduce matches the partial stage's split layout (row-quad at every length).
                if past + 8 > 0:
                    split_tokens = ((past + 8 + 1023) // 1024) * 32
                # row-oct split layout above T.
                if past + 8 > 16384:
                    split_tokens = ((past + 8 + 2047) // 2048) * 32
                splits = (past + 8 + split_tokens - 1) // split_tokens
                reduce_pair = iteration_block // 6
                h = iteration_block % 6
                if reduce_pair < 32:
                    row = reduce_pair // 4
                    kvhead = reduce_pair % 4
                    first = (row * 4 + kvhead) * splits
                    count = splits
                    head = kvhead * 6 + h
                    lane = tid % 64
                    ownmax = fx.Float32(-3.402823466e+38)
                    ownsum = fx.Float32(0)
                    if lane < count:
                        ownmax = fx.Float32(fx.ptr_load(maximum + (first + lane) * 6 + h))
                        ownsum = fx.Float32(fx.ptr_load(denominator + (first + lane) * 6 + h))
                    maxval = ownmax
                    for shift in (32, 16, 8, 4, 2, 1):
                        maxval = fx.max(maxval, maxval.shuffle_xor(shift, 64))
                    ownweight = fx.Float32(0)
                    if lane < count:
                        ownweight = fx.Float32(rocdl.exp2(T.f32, (ownmax - maxval).ir_value()))
                    total = ownsum * ownweight
                    for shift in (32, 16, 8, 4, 2, 1):
                        total = total + total.shuffle_xor(shift, 64)
                    divisor = fx.Float32(1)
                    if total > 0:
                        divisor = total
                    normalized_weight = ownweight / divisor
                    # up to 32 splits; issue every partial load, then the split-order FMA chain.
                    red_addr = fx.Int64(fx.ptrtoint(partial))
                    red_desc = fx.Vector.from_elements([fx.Int32(rocdl.readfirstlane(T.i32, red_addr.to(fx.Int32).ir_value())), fx.Int32(rocdl.readfirstlane(T.i32, red_addr.shrui(fx.Int64(32)).to(fx.Int32).ir_value())), fx.Int32(0x7FFFFFF0), fx.Int32(159744)], fx.Int32)
                    red_vals = []
                    for split in range_constexpr(32):
                        red_ok = (fx.Int32(split) < count).to(fx.Int32)
                        red_off = red_ok * ((((first + split) * 6 + h) * 256 + tid) * 2) + (1 - red_ok) * fx.Int32(0x7FFFFFF0)
                        red_vals.append(fx.Int16(llvm.call_intrinsic(T.i16, 'llvm.amdgcn.raw.buffer.load.i16', [red_desc.ir_value(), fx.Int32(red_off).ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value()], [], [])).bitcast(fx.BFloat16).to(fx.Float32))
                    value = fx.Float32(0)
                    for split in range_constexpr(32):
                        weight = gpu.shuffle_idx(normalized_weight, split, 64)
                        value = fx.Float32(fmath.fma(red_vals[split], weight, value))
                    if count > 32:
                        red_hi = []
                        for split_hi in range_constexpr(32):
                            red_ok = (fx.Int32(split_hi + 32) < count).to(fx.Int32)
                            red_off = red_ok * ((((first + split_hi + 32) * 6 + h) * 256 + tid) * 2) + (1 - red_ok) * fx.Int32(0x7FFFFFF0)
                            red_hi.append(fx.Int16(llvm.call_intrinsic(T.i16, 'llvm.amdgcn.raw.buffer.load.i16', [red_desc.ir_value(), fx.Int32(red_off).ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value()], [], [])).bitcast(fx.BFloat16).to(fx.Float32))
                        for split_hi in range_constexpr(32):
                            weight = gpu.shuffle_idx(normalized_weight, split_hi + 32, 64)
                            value = fx.Float32(fmath.fma(red_hi[split_hi], weight, value))
                    rounded = value.to(fx.BFloat16)
                    gateval = fx.Float32(fx.ptr_load(gate + row * 6144 + head * 256 + tid))
                    sigmoid = fx.Float32(1) / (fx.Float32(1) + fmath.exp2(-gateval * fx.Float32(1.442695041)))
                    final = (fx.Float32(rounded) * sigmoid).to(fx.BFloat16)
                    fx.ptr_store(final, gated + row * 6144 + head * 256 + tid)
                    amax = fmath.absf(fx.Float32(final))
                    for shift in range_constexpr(5):
                        amax = fx.max(amax, amax.shuffle_xor(1 << shift, 64))
                    bits_scale = (amax * fx.Float32(1 / 448)).bitcast(fx.Int32)
                    exponent = bits_scale.shrui(fx.Int32(23)) & fx.Int32(255)
                    if bits_scale & fx.Int32(8388607) != 0:
                        exponent = exponent + 1
                    exponent = fx.max(fx.Int32(1), fx.min(exponent, fx.Int32(254)))
                    scale = (exponent << fx.Int32(23)).bitcast(fx.Float32)
                    own = final.bitcast(fx.Int16).to(fx.Int32) & fx.Int32(65535)
                    peer = own.shuffle_xor(1, 64)
                    packed = own | peer << fx.Int32(16)
                    if tid % 2 == 0:
                        result = fx.Int32(llvm.inline_asm(T.i32, [packed.ir_value(), scale.ir_value(), fx.Int32(0).ir_value()], 'v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2', '=v,v,v,0', has_side_effects=True))
                        llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(out + row * 6144 + head * 256 + tid)).ir_value(), result.ir_value()], 'global_store_short $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                    if tid % 32 == 0:
                        qb = (head * 256 + tid) // 32
                        llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(scales + qb // 4 * 64 + qb % 4 * 16 + row)).ir_value(), fx.Int32(exponent).ir_value()], 'global_store_byte $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)
                if iteration_block == 0:
                    for s in range_constexpr(13):
                        i = tid + s * 256
                        if i < 3264:
                            if i >= 3072 or i % 16 >= 8:
                                llvm.InlineAsmOp(None, [fx.Uint64(fx.ptrtoint(scales + i)).ir_value(), fx.Int32(127).ir_value()], 'global_store_byte $0, $1, off sc0 sc1', 'v,v,~{memory}', has_side_effects=True)

            def head_stage(a, a_scale, weight, weight_scale, raw_output, head_tile):
                tid = iteration_tid
                (lane, wave) = (tid % 64, tid // 64)
                tile = head_tile
                reduction = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 0))
                demangle = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(reduction))

                def uniform(v):
                    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))

                def descriptor(p, size):
                    addr = fx.Int64(fx.ptrtoint(p))
                    return fx.Vector.from_elements([uniform(addr.to(fx.Int32)), uniform(addr.shrui(fx.Int64(32)).to(fx.Int32)), fx.Int32(size), fx.Int32(159744)], fx.Int32)

                def raw16(desc, offset, aux):
                    return fx.Vector(llvm.call_intrinsic(T.vec(4, T.i32), 'llvm.amdgcn.raw.buffer.load.v4i32', [desc.ir_value(), fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(aux).ir_value()], [], []))

                def raw4(desc, offset):
                    return fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.raw.buffer.load.i32', [desc.ir_value(), fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value()], [], []))

                def pin4(v):
                    return fx.Vector(llvm.inline_asm(T.vec(4, T.i32), [v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))

                def pin1(v):
                    return fx.Int32(llvm.inline_asm(T.i32, [v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))
                ad = descriptor(a, 40960)
                asd = descriptor(a_scale, 2752)
                bd = descriptor(weight, 635699200)
                bsd = descriptor(weight_scale, 39731200)
                alo = [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(2)]
                ahi = [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(2)]
                asc = [fx.Int32(0) for _ in range_constexpr(2)]
                bv = [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(12)]
                bs = [fx.Int32(0) for _ in range_constexpr(12)]
                acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
                for r in range_constexpr(3):
                    kb = r * 4 + wave
                    for atom in range_constexpr(4):
                        bv[r * 4 + atom] = raw16(bd, ((tile * 4 + atom) * 40 + kb) * 1024 + lane * 16, 2)
                        bs[r * 4 + atom] = raw4(bsd, (kb * 15520 + tile * 4 + atom) * 64 + (lane & fx.Int32(-4)))
                for r in range_constexpr(2):
                    kb = r * 4 + wave
                    off = lane % 16 * 5120 + kb * 128 + lane // 16 * 16
                    alo[r] = raw16(ad, off, 0)
                    ahi[r] = raw16(ad, off + 64, 0)
                    asc[r] = raw4(asd, kb * 64 + (lane & fx.Int32(-4)))
                for k in range_constexpr(10):
                    ar = k % 2
                    br = k % 3
                    lo = pin4(alo[ar])
                    hi = pin4(ahi[ar])
                    sa = pin1(asc[ar])
                    sa = sa.shrui((lane & fx.Int32(3)) * 8) & fx.Int32(255)
                    av = fx.Vector.from_elements([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]], fx.Int32)
                    for atom in range_constexpr(4):
                        b = pin4(bv[br * 4 + atom])
                        sb = pin1(bs[br * 4 + atom])
                        sb = sb.shrui((lane & fx.Int32(3)) * 8) & fx.Int32(255)
                        bf = fx.Vector.from_elements([b[0], b[1], b[2], b[3], fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0)], fx.Int32)
                        acc[atom] = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [av, bf, acc[atom], 0, 4, 0, sa, 0, sb]))
                    if fx.const_expr(k >= 1 and k + 1 < 10):
                        kb = (k + 1) * 4 + wave
                        off = lane % 16 * 5120 + kb * 128 + lane // 16 * 16
                        alo[(k + 1) % 2] = raw16(ad, off, 0)
                        ahi[(k + 1) % 2] = raw16(ad, off + 64, 0)
                        asc[(k + 1) % 2] = raw4(asd, kb * 64 + (lane & fx.Int32(-4)))
                    if fx.const_expr(k + 3 < 10):
                        kb = (k + 3) * 4 + wave
                        for atom in range_constexpr(4):
                            bv[br * 4 + atom] = raw16(bd, ((tile * 4 + atom) * 40 + kb) * 1024 + lane * 16, 2)
                            bs[br * 4 + atom] = raw4(bsd, (kb * 15520 + tile * 4 + atom) * 64 + (lane & fx.Int32(-4)))
                for atom in range_constexpr(4):
                    for e in range_constexpr(4):
                        fx.ptr_store(acc[atom][e], reduction + atom * 1024 + wave * 256 + lane * 4 + e)
                gpu.barrier()
                if wave == 0:
                    for atom in range_constexpr(4):
                        folded = [fx.Float32(0.0) for _ in range_constexpr(4)]
                        for e in range_constexpr(4):
                            off = atom * 1024 + lane * 4 + e
                            value = fx.ptr_load(reduction + off)
                            for w in range_constexpr(1, 4):
                                value = value + fx.ptr_load(reduction + off + w * 256)
                            folded[e] = value
                        rounded = fx.Vector.from_elements(folded, fx.Float32).to(fx.BFloat16)
                        for e in range_constexpr(4):
                            fx.ptr_store(rounded[e], demangle + atom * 320 + (lane // 16 * 4 + e) * 20 + lane % 16)
                gpu.barrier()
                if wave == 0:
                    row = lane // 4
                    col = lane % 4 * 4
                    if row < 8:
                        for atom in range_constexpr(4):
                            values = fx.ptr_load(demangle + atom * 320 + row * 20 + col, result_type=T.vec(4, T.bf16))
                            fx.ptr_store(values, raw_output + row * 248320 + tile * 64 + atom * 16 + col)
                gpu.barrier()

            def head_stream(a, a_scale, weight, weight_scale, raw_output, logits):
                # one weight stream across all of this CTA's tiles (5-slot LDS ring).
                tid = iteration_tid
                (lane, wave) = (tid % 64, tid // 64)

                def uniform(v):
                    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))

                def descriptor(p, size):
                    addr = fx.Int64(fx.ptrtoint(p))
                    return fx.Vector.from_elements([uniform(addr.to(fx.Int32)), uniform(addr.shrui(fx.Int64(32)).to(fx.Int32)), fx.Int32(size), fx.Int32(159744)], fx.Int32)

                def raw16(desc, offset, aux):
                    return fx.Vector(llvm.call_intrinsic(T.vec(4, T.i32), 'llvm.amdgcn.raw.buffer.load.v4i32', [desc.ir_value(), fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(aux).ir_value()], [], []))

                def raw4(desc, offset):
                    return fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.amdgcn.raw.buffer.load.i32', [desc.ir_value(), fx.Int32(offset).ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value()], [], []))

                def pin4(v):
                    return fx.Vector(llvm.inline_asm(T.vec(4, T.i32), [v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))

                def pin1(v):
                    return fx.Int32(llvm.inline_asm(T.i32, [v.ir_value()], '', '=&v,0,~{memory}', has_side_effects=True))

                def dma(desc, offset, dst):
                    llvm.InlineAsmOp(None, [uniform(dst).ir_value(), fx.Int32(offset).ir_value(), desc.ir_value()], 's_mov_b32 m0, $0\ns_nop 0\nbuffer_load_dwordx4 $1, $2, 0 offen nt lds', 's,v,s,~{memory},~{m0}', has_side_effects=True)

                def vmwait(n):
                    llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(%d)' % n, '~{memory}', has_side_effects=True)
                ad = descriptor(a, 40960)
                asd = descriptor(a_scale, 2752)
                bd = descriptor(weight, 635699200)
                bsd = descriptor(weight_scale, 39731200)
                ring = uniform(fx.ptrtoint(arena + 0)) + wave * 20480
                reduction = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 81920))
                demangle = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(reduction))
                ntiles = (3880 - iteration_block + 255) // 256
                alo = []
                ahi = []
                asc = []
                for k in range_constexpr(10):
                    kb = k * 4 + wave
                    off = lane % 16 * 5120 + kb * 128 + lane // 16 * 16
                    alo.append(raw16(ad, off, 0))
                    ahi.append(raw16(ad, off + 64, 0))
                    asc.append(raw4(asd, kb * 64 + (lane & fx.Int32(-4))))
                bs = [fx.Int32(0) for _ in range_constexpr(20)]
                for j in range_constexpr(5):
                    kb = j * 4 + wave
                    for atom in range_constexpr(4):
                        dma(bd, ((iteration_block * 4 + atom) * 40 + kb) * 1024 + lane * 16, ring + j * 4096 + atom * 1024)
                        bs[j * 4 + atom] = raw4(bsd, (kb * 15520 + iteration_block * 4 + atom) * 64 + (lane & fx.Int32(-4)))
                vmwait(40)
                alo = [pin4(x) for x in alo]
                ahi = [pin4(x) for x in ahi]
                asc = [pin1(x) for x in asc]
                for it in range(0, ntiles, 1):
                    tile = iteration_block + fx.Int32(it) * 256
                    nxt_on = (fx.Int32(it) + 1 < fx.Int32(ntiles)).to(fx.Int32)
                    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
                    for k in range_constexpr(10):
                        slot = k % 5
                        vmwait(32)
                        sa = asc[k].shrui((lane & fx.Int32(3)) * 8) & fx.Int32(255)
                        av = fx.Vector.from_elements([alo[k][0], alo[k][1], alo[k][2], alo[k][3], ahi[k][0], ahi[k][1], ahi[k][2], ahi[k][3]], fx.Int32)
                        bvals = [fx.Vector(fx.ptr_load(arena + wave * 20480 + slot * 4096 + atom * 1024 + lane * 16, result_type=T.vec(16, T.i8))).bitcast(fx.Int32) for atom in range_constexpr(4)]
                        sbs = [pin1(bs[slot * 4 + atom]) for atom in range_constexpr(4)]
                        rocdl.s_waitcnt(lgkmcnt=0)
                        if fx.const_expr(k < 5):
                            rt = tile
                            rkb = (k + 5) * 4 + wave
                            r_on = fx.Int32(1)
                        else:
                            rt = tile + 256
                            rkb = (k - 5) * 4 + wave
                            r_on = nxt_on
                        for atom in range_constexpr(4):
                            dma(bd, r_on * (((rt * 4 + atom) * 40 + rkb) * 1024 + lane * 16) + (1 - r_on) * fx.Int32(-16), ring + slot * 4096 + atom * 1024)
                            bs[slot * 4 + atom] = raw4(bsd, r_on * ((rkb * 15520 + rt * 4 + atom) * 64 + (lane & fx.Int32(-4))) + (1 - r_on) * fx.Int32(-16))
                        for atom in range_constexpr(4):
                            sb = sbs[atom].shrui((lane & fx.Int32(3)) * 8) & fx.Int32(255)
                            b = bvals[atom]
                            bf = fx.Vector.from_elements([b[0], b[1], b[2], b[3], fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0)], fx.Int32)
                            acc[atom] = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [av, bf, acc[atom], 0, 4, 0, sa, 0, sb]))
                    for atom in range_constexpr(4):
                        for e in range_constexpr(4):
                            fx.ptr_store(acc[atom][e], reduction + atom * 1024 + wave * 256 + lane * 4 + e)
                    gpu.barrier()
                    if wave == 0:
                        for atom in range_constexpr(4):
                            folded = [fx.Float32(0.0) for _ in range_constexpr(4)]
                            for e in range_constexpr(4):
                                off = atom * 1024 + lane * 4 + e
                                value = fx.ptr_load(reduction + off)
                                for w in range_constexpr(1, 4):
                                    value = value + fx.ptr_load(reduction + off + w * 256)
                                folded[e] = value
                            rounded = fx.Vector.from_elements(folded, fx.Float32).to(fx.BFloat16)
                            for e in range_constexpr(4):
                                fx.ptr_store(rounded[e], demangle + atom * 320 + (lane // 16 * 4 + e) * 20 + lane % 16)
                    gpu.barrier()
                    if wave == 0:
                        row = lane // 4
                        col = lane % 4 * 4
                        if row < 8:
                            for atom in range_constexpr(4):
                                values = fx.ptr_load(demangle + atom * 320 + row * 20 + col, result_type=T.vec(4, T.bf16))
                                fx.ptr_store(values, raw_output + row * 248320 + tile * 64 + atom * 16 + col)
                                fx.ptr_store(values.to(fx.Float32), logits + row * 248320 + tile * 64 + atom * 16 + col)
                    gpu.barrier()

            def copy_context_stage(copy_source, copy_target, copy_elements):
                copy_worker = (iteration_block - 192) * 256 + iteration_tid
                copy_chunks = copy_elements // 8
                for copy_chunk in range(copy_worker, copy_chunks, 16384):
                    copy_values = fx.Vector(fx.ptr_load(copy_source + copy_chunk * 8, result_type=T.vec(8, T.bf16)))
                    fx.ptr_store(copy_values, copy_target + copy_chunk * 8)
                if copy_worker < 8:
                    copy_tail = copy_chunks * 8 + copy_worker
                    if copy_tail < copy_elements:
                        fx.ptr_store(fx.ptr_load(copy_source + copy_tail), copy_target + copy_tail)

            def prepare_recurrent(raw, workspace, ba):
                index = iteration_block * 256 + iteration_tid
                if index < 8 * 6272:
                    row = index // 6272
                    column = index % 6272
                    value = fx.BFloat16(0.0)
                    if column < 6144:
                        value = fx.ptr_load(raw + row * 16512 + 10240 + column)
                    else:
                        value = fx.ptr_load(ba + row * 128 + column - 6144).to(fx.BFloat16)
                        fx.ptr_store(value, raw + row * 16512 + 10240 + column)
                    fx.ptr_store(value, workspace + row * 16512 + 10240 + column)

            def peer_handoff(epoch):
                head = iteration_block // 4
                split = iteration_block % 4
                peer = native + PEER_WORD + (head * 4 + split) * 16
                llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                gpu.barrier()
                if iteration_tid == 0:
                    system_store_u32(peer + 8, epoch)
                    llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                if iteration_tid < 64:
                    # one wave-wide poll covers all four peers (lane l -> peer l % 4).
                    failed = fx.Int32(rocdl.readfirstlane(T.i32, coherent_u32(native + 289).ir_value()))
                    ready = fx.Int32(0)
                    polls = fx.Int32(0)
                    while (ready == 0) & (failed == 0) & (polls < 1048576):
                        peer_epoch = coherent_u32(native + PEER_WORD + (head * 4 + iteration_tid % 4) * 16 + 8)
                        peer_ok = (peer_epoch >= epoch).to(fx.Int32)
                        peer_ok = fx.min(peer_ok, peer_ok.shuffle_xor(1, 64))
                        peer_ok = fx.min(peer_ok, peer_ok.shuffle_xor(2, 64))
                        ready = fx.Int32(rocdl.readfirstlane(T.i32, peer_ok.ir_value()))
                        failed = fx.Int32(rocdl.readfirstlane(T.i32, coherent_u32(native + 289).ir_value()))
                        llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                        polls = polls + 1
                    if iteration_tid == 0:
                        if (ready == 0) | (failed != 0):
                            fail_native()
                            failed = fx.Int32(2)
                        answer = fx.Int32(1)
                        if failed != 0:
                            answer = fx.Int32(2)
                        fx.ptr_store(answer, status_ptr + 4)
                gpu.barrier()
                # The following shared inverse values are private to this CTA/phase.
                inverse = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared, 16), fx.ptrtoint(arena + 8704))
                if fx.ptr_load(status_ptr + 4) == 1:
                    # lanes 8p + r load peer p's row-r square in one load; lane r sums in peer order.
                    gathered = fx.Int32(0)
                    if iteration_tid < 32:
                        gathered = coherent_u32(native + PEER_WORD + (head * 4 + iteration_tid // 8) * 16 + iteration_tid % 8)
                    peer_parts = [gathered] + [fx.Int32(gpu.shuffle_idx(gathered, iteration_tid % 64 + peer_split * 8, 64)) for peer_split in range_constexpr(1, 4)]
                    if iteration_tid < 8:
                        total = fx.Float32(0.0)
                        for peer_split in range_constexpr(4):
                            total = total + peer_parts[peer_split].bitcast(fx.Float32)
                        rms_input = fmath.fma(total, fx.Float32(1.0 / 128.0), fx.Float32(1e-06))
                        value = fx.Float32(llvm.inline_asm(T.f32, [rms_input.ir_value()], 'v_rsq_f32 $0, $1', '=v,v', has_side_effects=False))
                        fx.ptr_store(value, inverse + iteration_tid)
                gpu.barrier()
                return fx.Int32(fx.ptr_load(status_ptr + 4))

            def up_l2_touch(ut_first, ut_count, ut_steps, ut_k0=20):
                # pull Up k-steps 5 .. 5+ut_steps-1 of same-XCD Up CTAs (slots ut_first ..
                # +ut_count-1, block x + 8 slot) into this XCD's L2 (LDS-DMA, scratch arena 156672).
                ut_weight = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(14))
                ut_addr = fx.Int64(fx.ptrtoint(ut_weight))
                ut_lo = fx.Int32(rocdl.readfirstlane(T.i32, ut_addr.to(fx.Int32).ir_value()))
                ut_hi = fx.Int32(rocdl.readfirstlane(T.i32, ut_addr.shrui(fx.Int64(32)).to(fx.Int32).ir_value()))
                ut_desc = fx.Vector.from_elements([ut_lo, ut_hi, fx.Int32(89128960), fx.Int32(159744)], fx.Int32)
                ut_dst = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(arena + 156672) + iteration_tid // 64 * 256).ir_value()))
                ut_x = iteration_block % 8
                ut_lines = ut_steps * 32
                for ut_m in range_constexpr(ut_count):
                    ut_b = ut_x + (ut_first + ut_m) * 8
                    ut_up = ((ut_b > 0) & ((ut_b <= 136) | (ut_b >= 144))).to(fx.Int32)
                    ut_unit = ut_b - 1 - (ut_b >= 144).to(fx.Int32) * 7
                    ut_three = (ut_unit < 48).to(fx.Int32)
                    ut_gbase = ut_three * (ut_unit * 3) + (1 - ut_three) * (144 + (ut_unit - 48) * 2)
                    ut_groups = (2 + ut_three) * ut_up
                    for ut_p in range_constexpr((3 * 4 * ut_lines + 255) // 256):
                        ut_line = iteration_tid + ut_p * 256
                        ut_region = ut_line // ut_lines
                        ut_g = ut_region // 4
                        ut_on = (ut_g < ut_groups).to(fx.Int32)
                        ut_off = (((ut_gbase + ut_g) * 4 + ut_region % 4) * 40 + ut_k0) * 1024 + ut_line % ut_lines * 128
                        ut_off = ut_on * ut_off + (1 - ut_on) * fx.Int32(-128)
                        llvm.InlineAsmOp(None, [ut_dst.ir_value(), ut_off.ir_value(), ut_desc.ir_value()], 's_mov_b32 m0, $0\ns_nop 0\nbuffer_load_dword $1, $2, 0 offen lds', 's,v,s,~{memory},~{m0}', has_side_effects=True)

            def down_l2_touch():
                # k-steps 0 .. T-1 of the Down jobs of this XCD's six three-group Up CTAs.
                dt_weight = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(16))
                dt_addr = fx.Int64(fx.ptrtoint(dt_weight))
                dt_desc = fx.Vector.from_elements([fx.Int32(rocdl.readfirstlane(T.i32, dt_addr.to(fx.Int32).ir_value())), fx.Int32(rocdl.readfirstlane(T.i32, dt_addr.shrui(fx.Int64(32)).to(fx.Int32).ir_value())), fx.Int32(44564480), fx.Int32(159744)], fx.Int32)
                dt_dst = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(arena + 156672) + iteration_tid // 64 * 256).ir_value()))
                dt_x = iteration_block % 8
                dt_first = (dt_x == 0).to(fx.Int32)
                for dt_m in range_constexpr(6):
                    dt_blk = dt_x + (dt_first + dt_m) * 8
                    dt_job = fx.Int32((dt_blk + fx.Int32(dt_blk == 3).to(fx.Int32) * 56 + fx.Int32(dt_blk == 6).to(fx.Int32) * 56 + fx.Int32(dt_blk == 9).to(fx.Int32) * 40 + fx.Int32(dt_blk == 12).to(fx.Int32) * 40 + fx.Int32(dt_blk == 15).to(fx.Int32) * 40 + fx.Int32(dt_blk == 18).to(fx.Int32) * 32 + fx.Int32(dt_blk == 21).to(fx.Int32) * 32 + fx.Int32(dt_blk == 24).to(fx.Int32) * 32 + fx.Int32(dt_blk == 27).to(fx.Int32) * 40 + fx.Int32(dt_blk == 30).to(fx.Int32) * 40 + fx.Int32(dt_blk == 33).to(fx.Int32) * 32 + fx.Int32(dt_blk == 36).to(fx.Int32) * 32 + fx.Int32(dt_blk == 39).to(fx.Int32) * 32 + fx.Int32(dt_blk == 42).to(fx.Int32) * 16 + fx.Int32(dt_blk == 45).to(fx.Int32) * 16 + fx.Int32(dt_blk == 48).to(fx.Int32) * 16 + fx.Int32(dt_blk == 49).to(fx.Int32) * -40 + fx.Int32(dt_blk == 50).to(fx.Int32) * -32 + fx.Int32(dt_blk == 52).to(fx.Int32) * -40 + fx.Int32(dt_blk == 53).to(fx.Int32) * -32 + fx.Int32(dt_blk == 55).to(fx.Int32) * -40 + fx.Int32(dt_blk == 56).to(fx.Int32) * -32 + fx.Int32(dt_blk == 58).to(fx.Int32) * -16 + fx.Int32(dt_blk == 59).to(fx.Int32) * -56 + fx.Int32(dt_blk == 61).to(fx.Int32) * -16 + fx.Int32(dt_blk == 62).to(fx.Int32) * -56 + fx.Int32(dt_blk == 64).to(fx.Int32) * -16 + fx.Int32(dt_blk == 65).to(fx.Int32) * -32 + fx.Int32(dt_blk == 67).to(fx.Int32) * -40 + fx.Int32(dt_blk == 68).to(fx.Int32) * -32 + fx.Int32(dt_blk == 70).to(fx.Int32) * -40 + fx.Int32(dt_blk == 71).to(fx.Int32) * -32))
                    (dt_tile, dt_peer) = (dt_job // 3, dt_job % 3)
                    dt_start = dt_peer * 11 + fx.min(dt_peer, fx.Int32(1))
                    for dt_p in range_constexpr(4 * 128 // 256):
                        dt_line = iteration_tid + dt_p * 256
                        dt_r = dt_line // 128
                        dt_atom = dt_line % 128 // 32
                        dt_k = dt_line % 32 // 8
                        dt_off = ((dt_tile * 4 + dt_atom) * 136 + (dt_start + dt_r) * 4 + dt_k) * 1024 + dt_line % 8 * 128
                        llvm.InlineAsmOp(None, [dt_dst.ir_value(), dt_off.ir_value(), dt_desc.ir_value()], 's_mov_b32 m0, $0\ns_nop 0\nbuffer_load_dword $1, $2, 0 offen lds', 's,v,s,~{memory},~{m0}', has_side_effects=True)

            def relay_up_slices():
                # one CTA per XCD turns the 12 global Up slice sub-counters into 3 local words.
                if iteration_tid < 64:
                    rl_done = fx.Int32(0)
                    rl_failed = fx.Int32(0)
                    rl_polls = fx.Int32(0)
                    while (rl_done != 7) & (rl_failed == 0) & (rl_polls < 1048576):
                        rl_have = coherent_u32(native + CHUNK_WORD + 256 + (iteration_tid % 12) * 16)
                        rl_slice = iteration_tid % 12 // 4
                        rl_need = (fx.Int32(22) - (rl_slice == 0).to(fx.Int32) * 4) * (layer + 1)
                        rl_ok = (rl_have >= rl_need).to(fx.Int32)
                        rl_ok = fx.min(rl_ok, rl_ok.shuffle_xor(1, 64))
                        rl_ok = fx.min(rl_ok, rl_ok.shuffle_xor(2, 64))
                        rl_s0 = fx.Int32(rocdl.readfirstlane(T.i32, rl_ok.ir_value()))
                        rl_s1 = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(gpu.shuffle_idx(rl_ok, 4, 64)).ir_value()))
                        rl_s2 = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(gpu.shuffle_idx(rl_ok, 8, 64)).ir_value()))
                        rl_now = rl_s0 + rl_s1 * 2 + rl_s2 * 4
                        rl_new = rl_now & (fx.Int32(7) - rl_done)
                        if (rl_new != 0) & (iteration_tid == 0):
                            for rl_s in range_constexpr(3):
                                if (rl_new & (1 << rl_s)) != 0:
                                    fx.ptr_store(layer + 1, native + CHUNK_WORD + 1024 + (safe_xcc * 3 + rl_s) * 16)
                        rl_done = rl_done | rl_now
                        rl_failed = fx.Int32(rocdl.readfirstlane(T.i32, coherent_u32(native + 289).ir_value()))
                        llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                        rl_polls = rl_polls + 1
                    if (iteration_tid == 0) & (rl_done != 7):
                        fail_native()
                gpu.barrier()

            def relay_mixer_slices():
                # one CTA per XCD turns the 12 global mixer sub-counters into 3 local words.
                if iteration_tid < 64:
                    rm_done = fx.Int32(0)
                    rm_failed = fx.Int32(0)
                    rm_polls = fx.Int32(0)
                    while (rm_done != 7) & (rm_failed == 0) & (rm_polls < 1048576):
                        rm_have = coherent_u32(native + CHUNK_WORD + 640 + (iteration_tid % 12) * 16)
                        rm_ok = (rm_have >= (layer + 1) * 16).to(fx.Int32)
                        rm_ok = fx.min(rm_ok, rm_ok.shuffle_xor(1, 64))
                        rm_ok = fx.min(rm_ok, rm_ok.shuffle_xor(2, 64))
                        rm_s0 = fx.Int32(rocdl.readfirstlane(T.i32, rm_ok.ir_value()))
                        rm_s1 = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(gpu.shuffle_idx(rm_ok, 4, 64)).ir_value()))
                        rm_s2 = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(gpu.shuffle_idx(rm_ok, 8, 64)).ir_value()))
                        rm_now = rm_s0 + rm_s1 * 2 + rm_s2 * 4
                        rm_new = rm_now & (fx.Int32(7) - rm_done)
                        if (rm_new != 0) & (iteration_tid == 0):
                            for rm_s in range_constexpr(3):
                                if (rm_new & (1 << rm_s)) != 0:
                                    fx.ptr_store(layer + 1, native + CHUNK_WORD + 1408 + (safe_xcc * 3 + rm_s) * 16)
                        rm_done = rm_done | rm_now
                        rm_failed = fx.Int32(rocdl.readfirstlane(T.i32, coherent_u32(native + 289).ir_value()))
                        llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                        rm_polls = rm_polls + 1
                    if (iteration_tid == 0) & (rm_done != 7):
                        fail_native()
                gpu.barrier()

            def zero_mixer_reserve(mixer):
                worker = fx.Int32(0)
                if iteration_block != 0:
                    worker = iteration_block - 136
                for index in range(worker * 256 + iteration_tid, 40960, 8 * 256):
                    fx.ptr_store(fx.BFloat16(0.0), mixer + index)

            def fill_up_padding(scales):
                index = iteration_block * 256 + iteration_tid
                if index < 8896:
                    if (index >= 8704) | (index % 16 >= 8):
                        fx.ptr_store(fx.Uint8(127), scales + index)

            # Source-authored native phase positions; diagnostic sidework is explicit.
            hidden = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(0))
            residual = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(1))
            input_norm = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(2))
            input_normalized = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(18))
            residual1 = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(19))
            input_A = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(20))
            input_AS = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(21))
            combined_weight = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(4))
            combined_scale = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(5))
            raw = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(22))
            mixer = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(32))
            if alive == 1:
                # Delta BA clear ahead of the fold (system stores; covered by the fold counters).
                if (layer % 4 != 3) & (iteration_block < 8):
                    early_ba = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(25))
                    early_clear = iteration_block * 256 + iteration_tid
                    if early_clear < 1024:
                        system_store_u32(early_ba + early_clear, fx.Int32(0))
                if (iteration_block < 8) & (layer == 0):
                    saved_mode_0 = fx.Int32(llvm.inline_asm(T.i32, [], 's_getreg_b32 $0, 0xdc1', '=s,~{memory}', has_side_effects=True))
                    rms_stage(hidden, residual, input_norm, input_normalized, residual1, input_A, input_AS, iteration_block)
                    llvm.InlineAsmOp(None, [saved_mode_0.ir_value()], 's_waitcnt vmcnt(0) lgkmcnt(0)\ns_setreg_b32 0xdc1, $0', 's,~{memory}', has_side_effects=True)
                    gpu.barrier()
                    if iteration_tid == 0:
                        fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope='agent')
                        system_atomic_add_nw(native + CHUNK_WORD + 512 + iteration_block * 16)
                if (iteration_block < 64) & (layer > 0):
                    saved_mode_f = fx.Int32(llvm.inline_asm(T.i32, [], 's_getreg_b32 $0, 0xdc1', '=s,~{memory}', has_side_effects=True))
                    previous_partial = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(39 - 45))
                    alive = fx.max(alive, fold_chunk_stage(previous_partial, residual, input_norm, input_normalized, residual1, input_A, input_AS, epoch_base + 1))
                    llvm.InlineAsmOp(None, [saved_mode_f.ir_value()], 's_waitcnt vmcnt(0) lgkmcnt(0)\ns_setreg_b32 0xdc1, $0', 's,~{memory}', has_side_effects=True)
                    gpu.barrier()
                    if iteration_tid == 0:
                        system_atomic_add_nw(native + CHUNK_WORD + 512 + iteration_block // 8 * 16)
            mixer_A = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(30))
            mixer_AS = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(31))
            output_phase = fx.Int32(6)
            norm_phase = fx.Int32(8)
            if layer % 4 == 3:
                context = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(25))
                alive = fx.max(alive, attention_qgkv_stage(input_A, input_AS, combined_weight, combined_scale, raw, epoch_base + 1, 1))
                position = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16), table_address(8))
                qgamma = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(6))
                kgamma = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(7))
                query = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(23))
                gate = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(24))
                context_blocks = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16), table_address(9))
                if (alive == 1) & (iteration_block < 224):
                    attention_qk_stage(position, raw, qgamma, kgamma, query, gate, context, context_blocks)
                block_map = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16), table_address(10))
                tasks = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16), table_address(11))
                dispatch = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16), table_address(41))
                attn_partial = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(26))
                maximum = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(27))
                denominator = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(28))
                if alive == 1:
                    saved_attention_mode3 = fx.Int32(llvm.inline_asm(T.i32, [], 's_getreg_b32 $0, 0xdc1', '=s,~{memory}', has_side_effects=True))
                    # row-quad tasks for long contexts, row-pair tasks for short ones.
                    # per-wave row-quad partials at every context length.
                    # row-oct tasks above T tokens (half the tiles per task, each K / V tile read once).
                    if fx.Int32(fx.ptr_load(position)) + 8 > 16384:
                        attention_partial_oct_stage(query, context, block_map, tasks, dispatch, attn_partial, maximum, denominator, position)
                    else:
                        if fx.Int32(fx.ptr_load(position)) + 8 > 0:
                            attention_partial_stage(query, context, block_map, tasks, dispatch, attn_partial, maximum, denominator, position)
                        else:
                            attention_partial_pair(query, context, block_map, tasks, dispatch, attn_partial, maximum, denominator, position)
                    llvm.InlineAsmOp(None, [saved_attention_mode3.ir_value()], 's_waitcnt vmcnt(0) lgkmcnt(0)\ns_setreg_b32 0xdc1, $0', 's,~{memory}', has_side_effects=True)
                alive = fx.max(alive, native_grid(epoch_base + 4, 4))
                gated = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(29))
                mixer_A = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(30))
                mixer_AS = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(31))
                if (alive == 1) & (iteration_block < 192):
                    attention_reduce_stage(gate, tasks, dispatch, attn_partial, maximum, denominator, gated, mixer_A, mixer_AS, position)
                    # mixer_A slice published (slice head // 8, sub = row % 4).
                    llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)', '~{memory}', has_side_effects=True)
                    gpu.barrier()
                    if iteration_tid == 0:
                        mx_pair = iteration_block // 6
                        mx_head = mx_pair % 4 * 6 + iteration_block % 6
                        system_atomic_add_nw(native + CHUNK_WORD + 640 + (mx_head // 8 * 4 + mx_pair // 4 % 4) * 16)
                    if fx.const_expr(diagnostics):
                        clear_index = iteration_block * 256 + iteration_tid
                        if clear_index < 40960:
                            fx.ptr_store(fx.BFloat16(0.0), mixer + clear_index)
                output_phase = fx.Int32(5)
                norm_phase = fx.Int32(7)
            else:
                BA = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(25))
                conv_weight = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(6))
                conv_history = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(7))
                workspace = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(23))
                conv_final = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(24))
                alive = fx.max(alive, qkv_stage(input_A, input_AS, combined_weight, combined_scale, conv_weight, conv_history, raw, workspace, conv_final, BA, epoch_base + 1, 1))
                initial_state = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(8))
                A_log = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(9))
                dt_bias = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(10))
                core = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(26))
                final_state = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(27))
                squares = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(28))
                state_snapshots = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(41))
                head_norm = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(11))
                recurrent = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(29))
                mixer_A = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(30))
                mixer_AS = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(31))
                if (alive == 1) & (iteration_block < 192):
                    core_stage(raw, workspace, BA, initial_state, A_log, dt_bias, core, final_state, squares, state_snapshots)
                if (alive == 1) & (iteration_block >= 192):
                    core_tail_stage(raw, workspace, BA)
                    if (iteration_block >= 240) & (iteration_block < 248):
                        up_l2_touch(fx.Int32(0), 3, 5)
                if iteration_block < 192:
                    alive = fx.max(alive, peer_handoff(epoch_base + 4))
                if (alive == 1) & (iteration_block < 192):
                    saved_mode1 = fx.Int32(llvm.inline_asm(T.i32, [], 's_getreg_b32 $0, 0xdc1', '=s,~{memory}', has_side_effects=True))
                    finalize_stage(raw, core, squares, head_norm, recurrent, mixer_A, mixer_AS)
                    llvm.InlineAsmOp(None, [saved_mode1.ir_value()], 's_waitcnt vmcnt(0) lgkmcnt(0)\ns_setreg_b32 0xdc1, $0', 's,~{memory}', has_side_effects=True)
                    # mixer_A slice published (slice head // 16, sub = split).
                    gpu.barrier()
                    if iteration_tid == 0:
                        system_atomic_add_nw(native + CHUNK_WORD + 640 + (iteration_block // 64 * 4 + iteration_block % 4) * 16)
                    if fx.const_expr(diagnostics):
                        clear_index = iteration_block * 256 + iteration_tid
                        if clear_index < 40960:
                            fx.ptr_store(fx.BFloat16(0.0), mixer + clear_index)
                if fx.const_expr(diagnostics):
                    if (layer % 4 == 2) & (layer + 1 < 64):
                        if (alive == 1) & (iteration_block >= 192):
                            copy_target = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(45 + 25))
                            copy_source = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(45 + 43))
                            copy_elements = fx.Int32(table_address(45 + 44))
                            copy_context_stage(copy_source, copy_target, copy_elements)
                output_phase = fx.Int32(6)
                norm_phase = fx.Int32(8)
            output_weight = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(12))
            output_scale = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(13))
            if iteration_block >= 248:
                relay_mixer_slices()
                if (alive == 1) & (layer % 4 != 3):
                    up_l2_touch(fx.Int32(29), 3, 1, 16)
            alive = fx.max(alive, output_stage(mixer_A, mixer_AS, output_weight, output_scale, mixer, epoch_base + output_phase, output_phase))
            post_norm = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(3))
            post_normalized = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(33))
            residual2 = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), table_address(34))
            post_A = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(35))
            post_AS = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(36))
            up_weight = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(14))
            up_scale = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(15))
            activated = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(37))
            activated_AS = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(38))
            if (iteration_block > 0) & ((iteration_block <= 136) | (iteration_block >= 144)):
                alive = fx.max(alive, up_stage(post_A, post_AS, up_weight, up_scale, activated, activated_AS, epoch_base + norm_phase, norm_phase))
                if iteration_block >= 248:
                    relay_up_slices()
            else:
                norm_row = fx.Int32(0)
                if iteration_block != 0:
                    norm_row = iteration_block - 136
                # wait for all 240 output CTAs through the eight output counters.
                if iteration_tid < 64:
                    on_target = (layer + 1) * 30
                    on_ready = fx.Int32(0)
                    on_failed = fx.Int32(rocdl.readfirstlane(T.i32, coherent_u32(native + 289).ir_value()))
                    on_polls = fx.Int32(0)
                    while (on_ready == 0) & (on_failed == 0) & (on_polls < 1048576):
                        on_have = coherent_u32(native + CHUNK_WORD + 128 + iteration_tid % 8 * 16)
                        on_ok = (on_have >= on_target).to(fx.Int32)
                        for on_shift in (1, 2, 4):
                            on_ok = fx.min(on_ok, on_ok.shuffle_xor(on_shift, 64))
                        on_ready = fx.Int32(rocdl.readfirstlane(T.i32, on_ok.ir_value()))
                        on_failed = fx.Int32(rocdl.readfirstlane(T.i32, coherent_u32(native + 289).ir_value()))
                        llvm.InlineAsmOp(None, [], 's_sleep 1', '~{memory}', has_side_effects=True)
                        on_polls = on_polls + 1
                    if iteration_tid == 0:
                        on_answer = fx.Int32(1)
                        if (on_ready == 0) | (on_failed != 0):
                            fail_native()
                            on_answer = fx.Int32(2)
                        llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0)\nbuffer_inv sc1', '~{memory}', has_side_effects=True)
                        fx.ptr_store(on_answer, status_ptr + 3)
                gpu.barrier()
                alive = fx.max(alive, fx.Int32(fx.ptr_load(status_ptr + 3)))
                if (alive == 1) & (norm_row < 8):
                    saved_mode2 = fx.Int32(llvm.inline_asm(T.i32, [], 's_getreg_b32 $0, 0xdc1', '=s,~{memory}', has_side_effects=True))
                    rms_stage(mixer, residual1, post_norm, post_normalized, residual2, post_A, post_AS, norm_row)
                    llvm.InlineAsmOp(None, [saved_mode2.ir_value()], 's_waitcnt vmcnt(0) lgkmcnt(0)\ns_setreg_b32 0xdc1, $0', 's,~{memory}', has_side_effects=True)
                alive = fx.max(alive, native_postnorm(epoch_base + norm_phase, norm_phase, layer))
                if alive == 1:
                    down_l2_touch()
                if fx.const_expr(not diagnostics):
                    if alive == 1:
                        zero_mixer_reserve(mixer)
            if alive == 1:
                fill_up_padding(activated_AS)
            down_weight = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(16))
            down_scale = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), table_address(17))
            partial = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), table_address(39))
            alive = fx.max(alive, down_stage(activated, activated_AS, down_weight, down_scale, partial, epoch_base + norm_phase + 1, norm_phase + 1))
            if layer == 63:
                alive = fx.max(alive, native_grid(epoch_base + norm_phase + 2, norm_phase + 2, 1))
                # Native deferred final conversion precedes common external norm/head.
                head_hidden = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), endpoint_address(0))
                if (alive == 1) & (iteration_block < 160):
                    fold_stage(partial, head_hidden)
                alive = fx.max(alive, native_grid(command_base + 754, 10))
                head_residual = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), endpoint_address(1))
                final_norm = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), endpoint_address(2))
                head_normalized = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), endpoint_address(18))
                head_residual_out = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), endpoint_address(19))
                head_A = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), endpoint_address(20))
                head_AS = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), endpoint_address(21))
                if (alive == 1) & (iteration_block < 8):
                    saved_head_mode = fx.Int32(llvm.inline_asm(T.i32, [], 's_getreg_b32 $0, 0xdc1', '=s,~{memory}', has_side_effects=True))
                    rms_stage(head_hidden, head_residual, final_norm, head_normalized, head_residual_out, head_A, head_AS, iteration_block)
                    llvm.InlineAsmOp(None, [saved_head_mode.ir_value()], 's_waitcnt vmcnt(0) lgkmcnt(0)\ns_setreg_b32 0xdc1, $0', 's,~{memory}', has_side_effects=True)
                alive = fx.max(alive, native_grid(command_base + 755, 11))
                head_weight = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), endpoint_address(4))
                head_scale = fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, fx.AddressSpace.Global, 16), endpoint_address(5))
                head_raw = fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 16), endpoint_address(22))
                head_logits = fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), endpoint_address(32))
                if alive == 1:
                    head_stream(head_A, head_AS, head_weight, head_scale, head_raw, head_logits)
            llvm.InlineAsmOp(None, [], 's_waitcnt vmcnt(0) lgkmcnt(0)', '~{memory}', has_side_effects=True)
            gpu.barrier()
            if iteration_tid == 0:
                failed = coherent_u32(native + 289)
                if (alive != 1) | (failed != 0):
                    system_store_u32(control + 8, fx.Int32(2))
                    system_store_u32(control + 16 + iteration_block, fx.Int32(2))
                else:
                    system_store_u32(control + 16 + iteration_block, fx.Int32(1))
            layer = layer + 1

    @flyc.jit
    def launch(pointer_table: fx.Pointer, stream: fx.Stream):
        kernel(pointer_table).launch(grid=(256, 1, 1), block=(256, 1, 1), stream=stream)
    return launch
