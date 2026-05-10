//=============================================================================
// brnn.asm — Binary RNN inference for the C64.  KickAssembler.
//
// Build:    java -jar KickAss.jar brnn.asm -o brnn.prg
// Run:      LOAD"BRNN.PRG",8,1  : RUN
//=============================================================================

.const KERNAL_CHROUT = $FFD2
.const KERNAL_CLRSCR = $E544

// ---------- ZP allocation ---------------------------------------------------
.const wts_ptr   = $02   // 2B
.const thr_ptr   = $04   // 2B
.const sum_lo    = $06
.const sum_hi    = $07
.const out_idx   = $08
.const tmp1      = $09   // (also used as ptr lo for embed copy)
.const tmp2      = $0A   // (also used as ptr hi for embed copy)
.const rng_lo    = $0B
.const rng_hi    = $0C
.const cum_lo    = $0D
.const cum_hi    = $0E
.const min_pop   = $0F
.const tok       = $10

// sum_lo / sum_hi double as target_lo / target_hi during sampling
.const target_lo = sum_lo
.const target_hi = sum_hi

// ---------- memory map ------------------------------------------------------
.const POPLUT       = $1000
.const EXPLUT       = $1100
.const STATE_A      = $1200
.const STATE_B      = $1220
.const WEIGHTS_BUF  = $1240
.const POPCOUNTS    = $12C0
.const THRESHOLDS   = $1400
.const INITIAL_VEC  = $1C00
.const EMBED        = $1C20
.const HEAD         = $2420
.const FF_WEIGHTS   = $2C20

//=============================================================================
.pc = $0801 "BASIC stub"
:BasicUpstart2(start)

.pc = $0810 "Code"

start:
    sei
    lda #$36                 // BASIC out, KERNAL+I/O on
    sta $01

    lda #$0E                 // lower-case mode
    jsr KERNAL_CHROUT
    jsr KERNAL_CLRSCR

    // copy initial vector into state_a
    ldx #31
init_loop:
    lda INITIAL_VEC,x
    sta STATE_A,x
    dex
    bpl init_loop

    // seed RNG (don't allow 0)
    lda $D012
    ora #$01
    sta rng_lo
    lda $DC04
    ora #$01
    sta rng_hi

main_loop:
    jsr forward_pass         // populates POPCOUNTS for v = 0..127
    jsr sample_token         // returns sampled v in A
    sta tok
    jsr print_token
    lda tok
    jsr update_state
    jmp main_loop

//=============================================================================
// forward_pass: 4 FF layers + head computation
//=============================================================================
forward_pass:
    lda #<FF_WEIGHTS
    sta wts_ptr
    lda #>FF_WEIGHTS
    sta wts_ptr+1
    lda #<THRESHOLDS
    sta thr_ptr
    lda #>THRESHOLDS
    sta thr_ptr+1

    jsr ff_a_to_b
    jsr ff_b_to_a
    jsr ff_a_to_b
    jsr ff_b_to_a

    jmp head_compute

//=============================================================================
// One FF layer.  Inner loop does XOR-popcount of state row vs each weight
// column, compares against the precomputed cmp_val, sets bit in dst.
//=============================================================================
.macro ff_layer(src, dst) {
    lda #0
    sta out_idx
    ldx #31
!cl:
    sta dst,x
    dex
    bpl !cl-

!out:
    // popcount(src ^ wts_col_j)  -> sum_lo:sum_hi (16-bit)
    lda #0
    sta sum_lo
    sta sum_hi
    ldy #0
!ip:
    lda (wts_ptr),y
    eor src,y
    tax
    lda POPLUT,x
    clc
    adc sum_lo
    sta sum_lo
    bcc !nc+
    inc sum_hi
!nc:
    iny
    cpy #32
    bne !ip-

    // bit = (sum < cmp_val)?  Compare hi first, then lo.
    ldy #1
    lda sum_hi
    cmp (thr_ptr),y
    bcc !sb+
    bne !sk+
    ldy #0
    lda sum_lo
    cmp (thr_ptr),y
    bcc !sb+
!sk:
    jmp !co+

!sb:
    // set bit out_idx in dst
    lda out_idx
    lsr
    lsr
    lsr
    tax                      // X = byte index (0..31)
    lda out_idx
    and #$07
    tay                      // Y = bit position (0=MSB)
    lda bit_mask,y
    ora dst,x
    sta dst,x

!co:
    // wts_ptr += 32
    lda wts_ptr
    clc
    adc #32
    sta wts_ptr
    bcc !w+
    inc wts_ptr+1
!w:
    // thr_ptr += 2
    lda thr_ptr
    clc
    adc #2
    sta thr_ptr
    bcc !t+
    inc thr_ptr+1
!t:
    inc out_idx
    bne !out-                // 256 iterations
    rts
}

ff_a_to_b: ff_layer(STATE_A, STATE_B)
ff_b_to_a: ff_layer(STATE_B, STATE_A)

bit_mask:
    .byte $80, $40, $20, $10, $08, $04, $02, $01

//=============================================================================
// head_compute: for v=0..127, popcount(read ^ head[v]) -> POPCOUNTS[v]
// `read` is STATE_A bytes 16..31 (the read-half of the post-FF state).
//=============================================================================
head_compute:
    lda #<HEAD
    sta wts_ptr
    lda #>HEAD
    sta wts_ptr+1

    lda #0
    sta out_idx

hc_loop:
    lda #0
    sta sum_lo
    ldy #0
hc_inner:
    lda (wts_ptr),y
    eor STATE_A+16,y
    tax
    lda POPLUT,x
    clc
    adc sum_lo
    sta sum_lo
    iny
    cpy #16
    bne hc_inner

    ldx out_idx
    lda sum_lo
    sta POPCOUNTS,x

    lda wts_ptr
    clc
    adc #16
    sta wts_ptr
    bcc hc_nc
    inc wts_ptr+1
hc_nc:
    inc out_idx
    lda out_idx
    cmp #128
    bne hc_loop
    rts

//=============================================================================
// sample_token: temperature sampling using exp LUT.
//   1) min_pop = min(POPCOUNTS)
//   2) WEIGHTS_BUF[v] = EXPLUT[POPCOUNTS[v] - min_pop]   (8-bit)
//      target = sum of WEIGHTS_BUF                        (16-bit)
//   3) r = uniform random in [0, target)
//   4) scan WEIGHTS_BUF, subtract from r until r < weight; return that index
//=============================================================================
sample_token:
    // 1. find min popcount
    lda POPCOUNTS
    sta min_pop
    ldx #1
fm:
    lda POPCOUNTS,x
    cmp min_pop
    bcs fm_skip
    sta min_pop
fm_skip:
    inx
    cpx #128
    bne fm

    // 2. fill weights buffer + total
    lda #0
    sta target_lo
    sta target_hi
    ldx #0
wl:
    lda POPCOUNTS,x
    sec
    sbc min_pop              // delta in [0, 128]
    tay
    lda EXPLUT,y
    sta WEIGHTS_BUF,x
    clc
    adc target_lo
    sta target_lo
    bcc wl_nc
    inc target_hi
wl_nc:
    inx
    cpx #128
    bne wl

    // 3. rejection sample 16-bit r in [0, target)
rr:
    jsr rand16
    lda target_hi
    bne rr_big
    // target < 256
    lda rng_lo
    cmp target_lo
    bcs rr
    lda #0
    sta rng_hi
    jmp rr_done
rr_big:
    lda rng_hi
    cmp target_hi
    bcc rr_done
    bne rr
    lda rng_lo
    cmp target_lo
    bcs rr
rr_done:
    lda rng_lo
    sta cum_lo
    lda rng_hi
    sta cum_hi

    // 4. scan
    ldx #0
sc:
    lda WEIGHTS_BUF,x
    sta tmp1
    lda cum_hi
    bne sc_sub               // r >= 256, definitely > weight
    lda cum_lo
    cmp tmp1
    bcc sc_found             // r < weight, choose this v
sc_sub:
    lda cum_lo
    sec
    sbc tmp1
    sta cum_lo
    lda cum_hi
    sbc #0
    sta cum_hi
    inx
    cpx #128
    bne sc
    ldx #127                 // fallback (shouldn't reach)
sc_found:
    txa
    rts

//=============================================================================
// 16-bit Galois LFSR (poly 0xB400, period 65535)
//=============================================================================
rand16:
    lsr rng_hi
    ror rng_lo
    bcc r_no
    lda rng_hi
    eor #$B4
    sta rng_hi
r_no:
    rts

//=============================================================================
// update_state: STATE_A[16..31] := embed[A]
// (STATE_A[0..15] keeps the carry from the post-FF state.)
//=============================================================================
update_state:
    sta tmp1                 // tmp1 = token
    lsr
    lsr
    lsr
    lsr                      // A = token >> 4
    clc
    adc #>EMBED
    sta tmp2

    lda tmp1
    and #$0F
    asl
    asl
    asl
    asl                      // (token & 15) << 4
    clc
    adc #<EMBED
    sta tmp1
    bcc us_nc
    inc tmp2
us_nc:
    ldy #15
us_cp:
    lda (tmp1),y
    sta STATE_A+16,y
    dey
    bpl us_cp
    rts

//=============================================================================
// print_token: ASCII -> PETSCII (lower-case mode), then CHROUT.
//   * 0x0A (LF) -> 0x0D (CR)
//   * < 0x20 or >= 0x7F -> Skipped (control chars / extended ASCII)
//   * 'A'..'Z'  -> +0x80   (becomes shifted/uppercase)
//   * 'a'..'z'  -> -0x20   (becomes PETSCII lowercase)
//   * valid punctuation/numbers passed through directly
//=============================================================================
print_token:
    lda tok
    cmp #$0A          // LF -> CR
    bne pt_nlf
    lda #$0D
    jmp KERNAL_CHROUT
pt_nlf:
    cmp #$20          // <space ?
    bcc pt_skip
    cmp #$7F          // >=DEL ?
    bcs pt_skip
    cmp #$41          // <'A' ?
    bcc pt_dir
    cmp #$5B          // >'Z' ?
    bcs pt_lower
    ora #$80          // A-Z -> PETSCII shifted
    jmp KERNAL_CHROUT
pt_lower:
    cmp #$61          // <'a' ?
    bcc pt_dir
    cmp #$7B          // >'z' ?
    bcs pt_dir
    sec
    sbc #$20          // a-z -> PETSCII
pt_dir:
    jmp KERNAL_CHROUT
pt_skip:
    rts

//=============================================================================
// Data segments — KickAss pads gaps with zeros so we get one .prg.
//=============================================================================
.pc = POPLUT       "popcount LUT"  ; .import binary "poplut.bin"
.pc = EXPLUT       "exp LUT"       ; .import binary "explut.bin"
.pc = STATE_A      "state_a"       ; .fill 32, 0
.pc = STATE_B      "state_b"       ; .fill 32, 0
.pc = WEIGHTS_BUF  "weights_buf"   ; .fill 128, 0
.pc = POPCOUNTS    "popcounts"     ; .fill 128, 0
.pc = THRESHOLDS   "thresholds"    ; .import binary "thresholds.bin"
.pc = INITIAL_VEC  "initial"       ; .import binary "initial.bin"
.pc = EMBED        "embed"         ; .import binary "embed.bin"
.pc = HEAD         "head"          ; .import binary "head.bin"
.pc = FF_WEIGHTS   "ff_weights"    ; .import binary "ff_weights.bin"
