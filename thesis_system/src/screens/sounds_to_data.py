from __future__ import annotations

import streamlit as st

from src.components.ui import callout, route_button, section_title
from src.workflows.routes import go_to

# ── Title & Intro Header ──────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom:2rem;">
        <div style="font-family:var(--font-mono);font-size:0.85rem;color:var(--color-primary);font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem;">
            [EDUCATIONAL SIGNAL LAB : SOUNDS TO DATA]
        </div>
        <h1 style="font-size:2.4rem;font-weight:800;letter-spacing:-0.025em;color:var(--color-text-primary);margin-bottom:0.75rem;">
            Acoustic Signal Transcription &amp; Symbolic Tokenization
        </h1>
        <p style="font-size:1.05rem;color:var(--color-text-body);line-height:1.6;max-width:860px;">
            How raw acoustic field recordings of Cordillera flat-gong performances are converted into a discrete, machine-learnable vocabulary of <strong>12 compound rhythmic tokens</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 1. The 4-Phase Transcription Pipeline ────────────────────────────────────
section_title(
    "The 4-Stage Signal Transcription Pipeline",
    "Continuous acoustic audio is transformed into discrete symbolic tokens without manual score notation.",
)

c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    st.markdown(
        """
        <div class="pouls-card" style="height:100%;">
            <span class="pouls-badge pouls-badge-forest" style="margin-bottom:0.75rem;">STAGE 01</span>
            <div style="font-weight:700;font-size:1.05rem;margin-bottom:0.5rem;color:var(--color-text-primary);">Field Audio Capture</div>
            <p style="font-size:0.85rem;color:var(--color-text-body);line-height:1.5;margin:0;">
                Live acoustic field recording of community Sadanga Gangsa ensemble performances in Mountain Province (44.1 kHz, multi-channel).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="pouls-card" style="height:100%;">
            <span class="pouls-badge pouls-badge-forest" style="margin-bottom:0.75rem;">STAGE 02</span>
            <div style="font-weight:700;font-size:1.05rem;margin-bottom:0.5rem;color:var(--color-text-primary);">Onset Detection</div>
            <p style="font-size:0.85rem;color:var(--color-text-body);line-height:1.5;margin:0;">
                High-frequency content (HFC) and spectral flux tracking locate individual mallet strike impact points with millisecond precision.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="pouls-card" style="height:100%;">
            <span class="pouls-badge pouls-badge-forest" style="margin-bottom:0.75rem;">STAGE 03</span>
            <div style="font-weight:700;font-size:1.05rem;margin-bottom:0.5rem;color:var(--color-text-primary);">IOI Quantization</div>
            <p style="font-size:0.85rem;color:var(--color-text-body);line-height:1.5;margin:0;">
                Inter-Onset Intervals (IOI) are calculated and categorized into empirical timing bins: <strong>SHORT</strong> (&lt;280ms), <strong>MEDIUM</strong> (280 to 450ms), or <strong>LONG</strong> (&gt;450ms).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
        <div class="pouls-card" style="height:100%;">
            <span class="pouls-badge pouls-badge-forest" style="margin-bottom:0.75rem;">STAGE 04</span>
            <div style="font-weight:700;font-size:1.05rem;margin-bottom:0.5rem;color:var(--color-text-primary);">Compound Token</div>
            <p style="font-size:0.85rem;color:var(--color-text-body);line-height:1.5;margin:0;">
                Timing bin is paired with RMS onset velocity (<strong>WEAK</strong>, <strong>MEDIUM</strong>, <strong>STRONG</strong>) to yield standard syntax: <code>[GAP]_[STRENGTH]</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── 2. Interactive 12-Token Vocabulary Matrix with Web Audio Gong Synth ───────
section_title(
    "12-Token Vocabulary Matrix & Acoustic Simulation",
    "Explore every discrete token class in the verified 586-event corpus. Click Play to audition physical-modeled bronze gong chime acoustics.",
)

# Embedded Web Audio API synthesizer for instant in-browser audition
st.markdown(
    """
    <script>
    function playGangsaChime(strength, pitch) {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!window.__gangsaCtx) {
                window.__gangsaCtx = new AudioContext();
            }
            const ctx = window.__gangsaCtx;
            if (ctx.state === 'suspended') {
                ctx.resume();
            }
            const now = ctx.currentTime;
            const decay = strength === 'WEAK' ? 0.32 : (strength === 'MEDIUM' ? 0.70 : 1.35);
            const peak = strength === 'WEAK' ? 0.30 : (strength === 'MEDIUM' ? 0.55 : 0.85);

            const master = ctx.createGain();
            master.gain.setValueAtTime(0.0001, now);
            master.gain.linearRampToValueAtTime(peak, now + 0.012);
            master.gain.exponentialRampToValueAtTime(0.0001, now + decay);

            const filter = ctx.createBiquadFilter();
            filter.type = "bandpass";
            filter.frequency.setValueAtTime(strength === 'STRONG' ? 1420 : 920, now);
            filter.Q.setValueAtTime(3.8, now);

            [1.0, 1.414, 2.12, 2.76, 3.42].forEach((ratio, i) => {
                const osc = ctx.createOscillator();
                const g = ctx.createGain();
                osc.type = i % 2 === 0 ? "sine" : "triangle";
                osc.frequency.setValueAtTime(pitch * ratio, now);
                g.gain.setValueAtTime([1.0, 0.65, 0.45, 0.25, 0.15][i], now);
                g.gain.exponentialRampToValueAtTime(0.0001, now + (decay / (i + 1)));
                osc.connect(g);
                g.connect(filter);
                osc.start(now);
                osc.stop(now + decay + 0.1);
            });

            filter.connect(master);
            master.connect(ctx.destination);
        } catch(e) {
            console.error("Audio synth error:", e);
        }
    }
    </script>

    <div style="overflow-x:auto;">
    <table class="token-row-table">
        <thead>
            <tr>
                <th>Token Class</th>
                <th>Interval (IOI)</th>
                <th>Dynamic Velocity (RMS)</th>
                <th>Nominal Timing</th>
                <th>Musical Phrasing Function</th>
                <th style="text-align:center;">Interactive Audition</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><span class="token-badge">START_STRONG</span></td>
                <td>Initial (0 ms)</td>
                <td><strong style="color:var(--color-primary);">STRONG</strong> (High Peak)</td>
                <td>0 ms</td>
                <td>Accented opening stroke initiating phrase sequence</td>
                <td style="text-align:center;">
                    <button class="token-play-btn" onclick="playGangsaChime('STRONG', 440);">
                        &gt; Play
                    </button>
                </td>
            </tr>
            <tr>
                <td><span class="token-badge">START_MEDIUM</span></td>
                <td>Initial (0 ms)</td>
                <td>MEDIUM (Mid RMS)</td>
                <td>0 ms</td>
                <td>Standard opening phrase stroke</td>
                <td style="text-align:center;">
                    <button class="token-play-btn" onclick="playGangsaChime('MEDIUM', 440);">
                        &gt; Play
                    </button>
                </td>
            </tr>
            <tr>
                <td><span class="token-badge">START_WEAK</span></td>
                <td>Initial (0 ms)</td>
                <td>WEAK (Damped Low)</td>
                <td>0 ms</td>
                <td>Damped initial opening tap</td>
                <td style="text-align:center;">
                    <button class="token-play-btn" onclick="playGangsaChime('WEAK', 440);">
                        &gt; Play
                    </button>
                </td>
            </tr>
            <tr>
                <td><span class="token-badge">SHORT_STRONG</span></td>
                <td>Short (&lt; 280 ms)</td>
                <td><strong style="color:var(--color-primary);">STRONG</strong> (High Peak)</td>
                <td>~220 ms</td>
                <td>Rapid interlocking accented chime stroke</td>
                <td style="text-align:center;">
                    <button class="token-play-btn" onclick="playGangsaChime('STRONG', 520);">
                        &gt; Play
                    </button>
                </td>
            </tr>
            <tr>
                <td><span class="token-badge">SHORT_MEDIUM</span></td>
                <td>Short (&lt; 280 ms)</td>
                <td>MEDIUM (Mid RMS)</td>
                <td>~220 ms</td>
                <td>Rapid interlocking standard cadence stroke</td>
                <td style="text-align:center;">
                    <button class="token-play-btn" onclick="playGangsaChime('MEDIUM', 520);">
                        &gt; Play
                    </button>
                </td>
            </tr>
            <tr>
                <td><span class="token-badge">SHORT_WEAK</span></td>
                <td>Short (&lt; 280 ms)</td>
                <td>WEAK (Damped Low)</td>
                <td>~210 ms</td>
                <td>Rapid damped interlocking tap between pulses</td>
                <td style="text-align:center;">
                    <button class="token-play-btn" onclick="playGangsaChime('WEAK', 520);">
                        &gt; Play
                    </button>
                </td>
            </tr>
            <tr>
                <td><span class="token-badge">MEDIUM_STRONG</span></td>
                <td>Medium (280 to 450 ms)</td>
                <td><strong style="color:var(--color-primary);">STRONG</strong> (High Peak)</td>
                <td>~360 ms</td>
                <td>Accented cadential chime anchoring rhythmic cycle</td>
                <td style="text-align:center;">
                    <button class="token-play-btn" onclick="playGangsaChime('STRONG', 392);">
                        &gt; Play
                    </button>
                </td>
            </tr>
            <tr>
                <td><span class="token-badge">MEDIUM_MEDIUM</span></td>
                <td>Medium (280 to 450 ms)</td>
                <td>MEDIUM (Mid RMS)</td>
                <td>~350 ms</td>
                <td>Standard steady cadence pulse stroke</td>
                <td style="text-align:center;">
                    <button class="token-play-btn" onclick="playGangsaChime('MEDIUM', 392);">
                        &gt; Play
                    </button>
                </td>
            </tr>
            <tr>
                <td><span class="token-badge">MEDIUM_WEAK</span></td>
                <td>Medium (280 to 450 ms)</td>
                <td>WEAK (Damped Low)</td>
                <td>~340 ms</td>
                <td>Damped steady cadence tap maintaining meter</td>
                <td style="text-align:center;">
                    <button class="token-play-btn" onclick="playGangsaChime('WEAK', 392);">
                        &gt; Play
                    </button>
                </td>
            </tr>
            <tr>
                <td><span class="token-badge">LONG_STRONG</span></td>
                <td>Long (&gt; 450 ms)</td>
                <td><strong style="color:var(--color-primary);">STRONG</strong> (High Peak)</td>
                <td>~550 ms</td>
                <td>Extended open resonant chime marking phrase ending</td>
                <td style="text-align:center;">
                    <button class="token-play-btn" onclick="playGangsaChime('STRONG', 330);">
                        &gt; Play
                    </button>
                </td>
            </tr>
            <tr>
                <td><span class="token-badge">LONG_MEDIUM</span></td>
                <td>Long (&gt; 450 ms)</td>
                <td>MEDIUM (Mid RMS)</td>
                <td>~530 ms</td>
                <td>Extended phrase boundary ringing stroke</td>
                <td style="text-align:center;">
                    <button class="token-play-btn" onclick="playGangsaChime('MEDIUM', 330);">
                        &gt; Play
                    </button>
                </td>
            </tr>
            <tr>
                <td><span class="token-badge">LONG_WEAK</span></td>
                <td>Long (&gt; 450 ms)</td>
                <td>WEAK (Damped Low)</td>
                <td>~510 ms</td>
                <td>Phrase-end damped stroke releasing tension</td>
                <td style="text-align:center;">
                    <button class="token-play-btn" onclick="playGangsaChime('WEAK', 330);">
                        &gt; Play
                    </button>
                </td>
            </tr>
        </tbody>
    </table>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 3. Next Action Call-to-Action ─────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-top:2.5rem;padding:1.5rem;background:var(--color-surface);border:1px solid var(--color-border);display:flex;justify-content:space-between;align-items:center;">
        <div>
            <strong style="font-size:1.05rem;color:var(--color-text-primary);display:block;margin-bottom:0.25rem;">Ready to benchmark algorithms on this token vocabulary?</strong>
            <span style="font-size:0.9rem;color:var(--color-text-muted);">Proceed to Track A to inspect the verified 586-event corpus and configure LORO cross-validation.</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c_back, c_next = st.columns([1, 2.5])
with c_back:
    if st.button("<- Return to Overview", key="sounds_back_home", type="secondary", use_container_width=True):
        go_to("home")
with c_next:
    if st.button("Proceed to Track A: Step 1 (Upload Data) ->", key="sounds_to_track_a", type="primary", use_container_width=True):
        go_to("compare_data")
