import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
from collections import deque


# === BOUT DETECTION CLASS ===
class BoutDetector:
    """
    Real-time bout detection using Burgues' method with second derivative.
    """

    def __init__(self, fs=100, hl=0.3, ampthresh=10.0, sd_thr=0.0, freq_window=1.0):
        """
        Args:
            fs: Sampling frequency (Hz)
            hl: Half-life time for EWMA filter (seconds)
            ampthresh: Amplitude threshold for bout filtering
            sd_thr: Threshold for second derivative (typically 0)
            freq_window: Time window for bout frequency calculation (seconds)
        """
        self.fs = fs
        self.hl = hl
        self.ampthresh = ampthresh
        self.sd_thr = sd_thr
        self.freq_window = freq_window

        # Calculate EWMA alpha from half-life
        self.alpha = 1 - np.exp(-np.log(2) / (hl * fs))
        self.alpha_derivative = 1 - np.exp(-np.log(2) / (hl * fs))

        # State for EWMA filters
        self.s_prev = None
        self.sd_prev = None
        self.sdd_prev = None
        self.s_prev_for_deriv = None
        self.sd_prev_for_deriv = None

        # Bout state machine
        self.in_bout = False
        self.bout_start_time = None
        self.bout_start_idx = None
        self.bout_count = 0

        # Store first derivative values during current bout
        self.current_bout_derivatives = []

        self.bout_times = []
        self.bout_regions = []

    def reset(self):
        """Reset all state for reprocessing"""
        self.s_prev = None
        self.sd_prev = None
        self.sdd_prev = None
        self.s_prev_for_deriv = None
        self.sd_prev_for_deriv = None
        self.in_bout = False
        self.bout_start_time = None
        self.bout_start_idx = None
        self.bout_count = 0
        self.current_bout_derivatives = []
        self.bout_times = []
        self.bout_regions = []

    def update_parameters(self, hl=None, ampthresh=None, sd_thr=None):
        """Update parameters on the fly"""
        if hl is not None:
            self.hl = hl
            self.alpha = 1 - np.exp(-np.log(2) / (hl * self.fs))
            self.alpha_derivative = 1 - np.exp(-np.log(2) / (hl * self.fs))
        if ampthresh is not None:
            self.ampthresh = ampthresh
        if sd_thr is not None:
            self.sd_thr = sd_thr

    def ewma_update(self, new_value, prev_value, alpha):
        if prev_value is None:
            return new_value
        return alpha * new_value + (1 - alpha) * prev_value

    def process_sample(self, raw_value, current_time, sample_idx):
        """
        Process one sample through Burgues' bout detection pipeline.

        Returns: (smoothed_signal, first_derivative, second_derivative, bout_detected, bout_amplitude)
        """

        # Step 1: Smooth the raw signal
        xs = self.ewma_update(raw_value, self.s_prev, self.alpha)
        self.s_prev = xs

        bout_detected = False
        bout_amplitude = 0
        x_prime_s = 0
        x_double_prime_s = 0

        # Step 2: Compute FIRST derivative from smoothed signal
        if self.s_prev_for_deriv is not None:
            x_prime_raw = self.fs * (xs - self.s_prev_for_deriv)

            # Step 3: Smooth the FIRST derivative
            x_prime_s = self.ewma_update(x_prime_raw, self.sd_prev, self.alpha_derivative)
            self.sd_prev = x_prime_s

            # Step 4: Compute SECOND derivative from smoothed first derivative
            if self.sd_prev_for_deriv is not None:
                x_double_prime_raw = self.fs * (x_prime_s - self.sd_prev_for_deriv)

                # Step 5: Smooth the SECOND derivative
                x_double_prime_s = self.ewma_update(x_double_prime_raw, self.sdd_prev, self.alpha_derivative)
                self.sdd_prev = x_double_prime_s

                # Step 6: Bout detection based on SMOOTHED second derivative zero crossings
                is_rising = x_double_prime_s > self.sd_thr

                if is_rising and not self.in_bout:
                    # FIRST CROSSING: bout STARTS
                    self.in_bout = True
                    self.bout_start_time = current_time
                    self.bout_start_idx = sample_idx
                    self.current_bout_derivatives = [x_prime_s]

                elif is_rising and self.in_bout:
                    # DURING bout: continue collecting first derivative values
                    self.current_bout_derivatives.append(x_prime_s)

                elif not is_rising and self.in_bout:
                    # SECOND CROSSING: bout ENDS
                    self.in_bout = False
                    bout_end_time = current_time
                    bout_end_idx = sample_idx

                    if len(self.current_bout_derivatives) > 0:
                        # Calculate amplitude = peak-to-peak of FIRST derivative during bout
                        max_deriv = max(self.current_bout_derivatives)
                        min_deriv = min(self.current_bout_derivatives)
                        amp = max_deriv - min_deriv
                        bout_amplitude = amp

                        # Determine if bout is accepted based on amplitude threshold
                        accepted = amp > self.ampthresh

                        # Store bout region for visualization
                        self.bout_regions.append({
                            'start_time': self.bout_start_time,
                            'end_time': bout_end_time,
                            'start_idx': self.bout_start_idx,
                            'end_idx': bout_end_idx,
                            'amplitude': amp,
                            'accepted': accepted,
                            'max_deriv': max_deriv,
                            'min_deriv': min_deriv,
                            'mean_deriv': np.mean(self.current_bout_derivatives),
                        })

                        if accepted:
                            self.bout_count += 1
                            bout_detected = True
                            self.bout_times.append(current_time)

                    # Clear for next bout
                    self.current_bout_derivatives = []

                # Store previous smoothed first derivative for next second derivative calculation
                self.sd_prev_for_deriv = x_prime_s
            else:
                # Initialize on first iteration
                self.sd_prev_for_deriv = x_prime_s

        # Update previous smoothed signal for next first derivative calculation
        self.s_prev_for_deriv = xs

        return xs, x_prime_s, x_double_prime_s, bout_detected, bout_amplitude

    def compute_bout_frequency(self, times, current_ampthresh):
        """
        Compute bout frequency at each time point using a sliding window.

        Args:
            times: Array of time values
            current_ampthresh: Current amplitude threshold for filtering bouts

        Returns:
            Array of bout frequencies (Hz) at each time point
        """
        bout_freq = np.zeros_like(times)

        # Get accepted bout times based on current threshold
        accepted_bout_times = [r['start_time'] for r in self.bout_regions
                               if r['amplitude'] > current_ampthresh]

        if len(accepted_bout_times) == 0:
            return bout_freq

        accepted_bout_times = np.array(accepted_bout_times)

        # For each time point, count bouts in the preceding window
        for i, t in enumerate(times):
            window_start = t - self.freq_window
            bouts_in_window = np.sum((accepted_bout_times >= window_start) &
                                     (accepted_bout_times <= t))
            bout_freq[i] = bouts_in_window / self.freq_window

        return bout_freq


# === OFFLINE DASHBOARD ===
class OfflineDashboard:
    def __init__(self, csv_file):
        # Load data
        print(f"Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file)

        # Extract columns
        self.times_raw = self.df['Time'].values

        # Convert from Unix epoch to elapsed time starting at 0
        self.times = (self.times_raw - self.times_raw[0])/1000000
        self.gas_raw = self.df['Gas'].values

        print(f"Loaded {len(self.times)} samples")
        print(f"Duration: {self.times[-1] - self.times[0]:.1f} seconds")

        # Initialize bout detector
        self.bout_detector = BoutDetector(fs=100, hl=0.1, ampthresh=2.1, sd_thr=0.0, freq_window=1)

        # Process data initially
        self.reprocess_data()

        # Create figure
        self.create_figure()

    def reprocess_data(self):
        """Reprocess all data with current parameters"""
        print(
            f"Processing with hl={self.bout_detector.hl:.2f}, ampthresh={self.bout_detector.ampthresh:.1f}, sd_thr={self.bout_detector.sd_thr:.1f}")

        self.bout_detector.reset()

        self.gas_smoothed = np.zeros_like(self.gas_raw)
        self.first_deriv = np.zeros_like(self.gas_raw)
        self.second_deriv = np.zeros_like(self.gas_raw)
        self.bout_markers = np.zeros_like(self.gas_raw)
        self.bout_counts = np.zeros_like(self.gas_raw)

        for i, (t, gas) in enumerate(zip(self.times, self.gas_raw)):
            xs, xps, xdds, bout_detected, bout_amp = self.bout_detector.process_sample(gas, t, i)

            self.gas_smoothed[i] = xs
            self.first_deriv[i] = xps
            self.second_deriv[i] = xdds
            self.bout_markers[i] = 1 if bout_detected else 0
            self.bout_counts[i] = self.bout_detector.bout_count

        print(f"  -> Found {self.bout_detector.bout_count} accepted bouts")
        print(f"  -> Total bout windows: {len(self.bout_detector.bout_regions)}")

    def create_figure(self):
        self.fig = plt.figure(figsize=(16, 12))
        gs = self.fig.add_gridspec(6, 2, hspace=0.35, wspace=0.3,
                                   left=0.08, right=0.95, top=0.93, bottom=0.08)

        # Plot axes
        self.ax_gas = self.fig.add_subplot(gs[0, :])
        self.ax_smoothed = self.fig.add_subplot(gs[1, :])
        self.ax_derivative = self.fig.add_subplot(gs[2, :])
        self.ax_second_derivative = self.fig.add_subplot(gs[3, :])
        self.ax_bout_freq = self.fig.add_subplot(gs[4, :])  # Changed from bout_count
        self.ax_amplitude_scatter = self.fig.add_subplot(gs[5, 0])
        self.ax_stats = self.fig.add_subplot(gs[5, 1])

        # Gas plot (raw)
        self.line_gas, = self.ax_gas.plot([], [], 'b-', linewidth=0.5, alpha=0.5, label='Raw Gas')
        self.ax_gas.set_ylabel('Gas (Raw)')
        self.ax_gas.set_ylim(-100, 100)
        self.ax_gas.set_title('Raw Gas Sensor Signal', fontweight='bold', fontsize=10)
        self.ax_gas.legend(loc='upper right')
        self.ax_gas.grid(True, alpha=0.3)

        # Smoothed gas plot
        self.line_smoothed, = self.ax_smoothed.plot([], [], 'darkblue', linewidth=1.5, label='Smoothed Gas')
        self.scatter_bouts_smoothed = self.ax_smoothed.scatter([], [], c='red', s=50, zorder=5, label='Bouts')
        self.ax_smoothed.set_ylabel('Gas (Smoothed)')
        self.ax_smoothed.set_ylim(-50, 100)
        self.ax_smoothed.set_title('EWMA-Smoothed Gas Signal', fontweight='bold', fontsize=10)
        self.ax_smoothed.legend(loc='upper right')
        self.ax_smoothed.grid(True, alpha=0.3)

        # First derivative plot
        self.line_derivative, = self.ax_derivative.plot([], [], 'g-', linewidth=1.5, label="Smoothed x's (1st deriv)",
                                                        zorder=10)
        self.bout_patches_deriv = []
        self.ax_derivative.set_ylabel("x's (1st Derivative)")
        self.ax_derivative.set_ylim(-20, 30)
        self.ax_derivative.set_title("Smoothed First Derivative - Bout Segments Highlighted", fontweight='bold',
                                     fontsize=10)
        self.ax_derivative.legend(loc='upper right')
        self.ax_derivative.grid(True, alpha=0.3)

        # Second derivative plot
        self.line_second_derivative, = self.ax_second_derivative.plot([], [], 'purple', linewidth=1.5,
                                                                      label="Smoothed x''s (2nd deriv)", zorder=10)
        self.threshold_line = self.ax_second_derivative.axhline(y=0, color='orange', linestyle='--', linewidth=2,
                                                                label='sd_thr')
        self.bout_patches_second = []
        self.ax_second_derivative.set_ylabel("x''s (2nd Derivative)")
        self.ax_second_derivative.set_ylim(-50, 50)
        self.ax_second_derivative.set_title(
            "Smoothed Second Derivative - Bouts = Positive Excursions (Green=Accept, Red=Reject)", fontweight='bold',
            fontsize=10)
        self.ax_second_derivative.legend(loc='upper right')
        self.ax_second_derivative.grid(True, alpha=0.3)

        # Bout frequency (CHANGED FROM BOUT COUNT)
        self.line_bout_freq, = self.ax_bout_freq.plot([], [], 'purple', linewidth=2, label='Bout Frequency')
        self.ax_bout_freq.set_ylabel('Bout Frequency (Hz)')
        self.ax_bout_freq.set_xlabel('Time (s)')
        self.ax_bout_freq.set_ylim(0, 2.0)  # Adjust as needed
        self.ax_bout_freq.set_title('Bout Frequency (1-second sliding window)', fontweight='bold', fontsize=10)
        self.ax_bout_freq.legend(loc='upper left')
        self.ax_bout_freq.grid(True, alpha=0.3)

        # Amplitude scatter
        self.scatter_accepted = self.ax_amplitude_scatter.scatter([], [], c='green', s=50,
                                                                  alpha=0.6, label='Accepted', edgecolors='black')
        self.scatter_rejected = self.ax_amplitude_scatter.scatter([], [], c='red', s=50,
                                                                  alpha=0.6, label='Rejected', edgecolors='black')
        self.amp_threshold_line = self.ax_amplitude_scatter.axhline(y=0, color='blue',
                                                                    linestyle='--', linewidth=2, label='Threshold')
        self.ax_amplitude_scatter.set_xlabel('Time (s)')
        self.ax_amplitude_scatter.set_ylabel('Bout Amplitude (peak-to-peak of x\'s)')
        self.ax_amplitude_scatter.set_ylim(0, 100)
        self.ax_amplitude_scatter.set_title('Bout Amplitudes', fontweight='bold', fontsize=10)
        self.ax_amplitude_scatter.legend(loc='upper left')
        self.ax_amplitude_scatter.grid(True, alpha=0.3)

        # Stats
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.05, 0.95, '', transform=self.ax_stats.transAxes,
                                             verticalalignment='top', fontfamily='monospace',
                                             fontsize=9)

        # Sliders
        slider_height = 0.012
        slider_spacing = 0.016
        slider_left = 0.15
        slider_width = 0.3
        base_y = 0.015

        ax_hl = plt.axes([slider_left, base_y + 2 * slider_spacing, slider_width, slider_height])
        self.slider_hl = Slider(ax_hl, 'Half-life', 0, 1.0,
                                valinit=self.bout_detector.hl, valstep=0.05)

        ax_amp = plt.axes([slider_left, base_y + slider_spacing, slider_width, slider_height])
        self.slider_amp = Slider(ax_amp, 'Amp Thresh', 0, 100,
                                 valinit=self.bout_detector.ampthresh, valstep=0.05)

        ax_sd = plt.axes([slider_left, base_y, slider_width, slider_height])
        self.slider_sd = Slider(ax_sd, 'SD Thresh', -10, 10,
                                valinit=self.bout_detector.sd_thr, valstep=0.5)

        ax_reprocess = plt.axes([slider_left + slider_width + 0.05, base_y + slider_spacing, 0.1, 0.025])
        self.btn_reprocess = Button(ax_reprocess, 'Reprocess')

        # Connect callbacks
        self.slider_hl.on_changed(self.on_slider_change)
        self.slider_amp.on_changed(self.on_slider_change)
        self.slider_sd.on_changed(self.on_slider_change)
        self.btn_reprocess.on_clicked(self.on_reprocess)

        # Initial plot
        self.update_plots()

    def on_slider_change(self, val):
        """Just update the plots, don't reprocess yet"""
        self.update_plots()

    def on_reprocess(self, event):
        """Reprocess data with new parameters"""
        self.bout_detector.update_parameters(
            hl=self.slider_hl.val,
            ampthresh=self.slider_amp.val,
            sd_thr=self.slider_sd.val
        )
        self.reprocess_data()
        self.update_plots()

    def update_plots(self):
        """Update all plots"""

        # Update basic lines
        self.line_gas.set_data(self.times, self.gas_raw)
        self.line_smoothed.set_data(self.times, self.gas_smoothed)
        self.line_derivative.set_data(self.times, self.first_deriv)
        self.line_second_derivative.set_data(self.times, self.second_deriv)

        # Compute and plot bout frequency instead of bout count
        current_ampthresh = self.slider_amp.val
        bout_freq = self.bout_detector.compute_bout_frequency(self.times, current_ampthresh)
        self.line_bout_freq.set_data(self.times, bout_freq)

        # Update y-limit based on actual max frequency
        if len(bout_freq) > 0 and np.max(bout_freq) > 0:
            max_freq = np.max(bout_freq)
            self.ax_bout_freq.set_ylim(0, max(max_freq * 1.2, 0.5))  # At least 0.5 Hz range

        # Update bout markers on smoothed gas
        bout_indices = np.where(self.bout_markers == 1)[0]
        if len(bout_indices) > 0:
            bout_times = self.times[bout_indices]
            bout_gas = self.gas_smoothed[bout_indices]
            self.scatter_bouts_smoothed.set_offsets(np.c_[bout_times, bout_gas])
        else:
            self.scatter_bouts_smoothed.set_offsets(np.empty((0, 2)))

        # Clear old bout patches on first derivative
        for patch in self.bout_patches_deriv:
            patch.remove()
        self.bout_patches_deriv.clear()

        # Clear old bout patches on second derivative
        for patch in self.bout_patches_second:
            patch.remove()
        self.bout_patches_second.clear()

        # Draw bout regions
        for region in self.bout_detector.bout_regions:
            # Re-evaluate acceptance based on current slider value
            accepted = region['amplitude'] > current_ampthresh
            color = 'red' if accepted else 'darkred'

            # Extract the derivative segment during this bout
            start_idx = region['start_idx']
            end_idx = region['end_idx']

            bout_times_segment = self.times[start_idx:end_idx + 1]
            bout_deriv_segment = self.first_deriv[start_idx:end_idx + 1]

            if len(bout_times_segment) > 1:
                # Draw RED line over the green line for bout segment
                line, = self.ax_derivative.plot(
                    bout_times_segment,
                    bout_deriv_segment,
                    color=color,
                    linewidth=3,
                    alpha=0.9,
                    zorder=15
                )
                self.bout_patches_deriv.append(line)

                # Add amplitude label
                mid_time = (bout_times_segment[0] + bout_times_segment[-1]) / 2
                max_val = max(bout_deriv_segment)
                text_color = 'green' if accepted else 'red'
                text1 = self.ax_derivative.text(
                    mid_time,
                    max_val + 5,
                    f"{region['amplitude']:.1f}",
                    ha='center', va='bottom',
                    fontsize=9,
                    color='white',
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.4',
                        facecolor=text_color,
                        edgecolor='black',
                        linewidth=2,
                        alpha=0.9
                    ),
                    zorder=20
                )
                self.bout_patches_deriv.append(text1)

            # Draw rectangle on second derivative plot
            width = region['end_time'] - region['start_time']
            rect_color = 'green' if accepted else 'red'
            alpha_fill = 0.3 if accepted else 0.2

            rect2 = Rectangle((region['start_time'], -50),
                              width, 100,
                              facecolor=rect_color, edgecolor=rect_color,
                              alpha=alpha_fill, linewidth=1.5)
            self.ax_second_derivative.add_patch(rect2)
            self.bout_patches_second.append(rect2)

        # Update amplitude scatter
        if len(self.bout_detector.bout_regions) > 0:
            accepted_times = [r['start_time'] for r in self.bout_detector.bout_regions if
                              r['amplitude'] > current_ampthresh]
            accepted_amps = [r['amplitude'] for r in self.bout_detector.bout_regions if
                             r['amplitude'] > current_ampthresh]
            rejected_times = [r['start_time'] for r in self.bout_detector.bout_regions if
                              r['amplitude'] <= current_ampthresh]
            rejected_amps = [r['amplitude'] for r in self.bout_detector.bout_regions if
                             r['amplitude'] <= current_ampthresh]

            self.scatter_accepted.set_offsets(
                np.c_[accepted_times, accepted_amps] if accepted_times else np.empty((0, 2)))
            self.scatter_rejected.set_offsets(
                np.c_[rejected_times, rejected_amps] if rejected_times else np.empty((0, 2)))

            # Update threshold line
            self.amp_threshold_line.set_ydata([current_ampthresh, current_ampthresh])

        # Update threshold line on second derivative
        self.threshold_line.set_ydata([self.slider_sd.val, self.slider_sd.val])

        # Update stats
        total_bouts = len(self.bout_detector.bout_regions)
        accepted_bouts = sum(1 for r in self.bout_detector.bout_regions if r['amplitude'] > current_ampthresh)
        rejected_bouts = total_bouts - accepted_bouts

        # Calculate frequency statistics
        if len(bout_freq) > 0:
            max_bout_freq = np.max(bout_freq)
            mean_bout_freq = np.mean(bout_freq)
        else:
            max_bout_freq = mean_bout_freq = 0

        if len(self.bout_detector.bout_regions) > 0:
            all_amps = [r['amplitude'] for r in self.bout_detector.bout_regions]
            min_amp = min(all_amps)
            max_amp = max(all_amps)
            mean_amp = np.mean(all_amps)
        else:
            min_amp = max_amp = mean_amp = 0

        stats_str = f"""OFFLINE ANALYSIS

Duration:      {self.times[-1] - self.times[0]:8.1f} s
Samples:       {len(self.times):8d}

BOUT ANALYSIS
Total Windows: {total_bouts:8d}
Accepted:      {accepted_bouts:8d}
Rejected:      {rejected_bouts:8d}

BOUT FREQUENCY
Max Freq:      {max_bout_freq:8.2f} Hz
Mean Freq:     {mean_bout_freq:8.2f} Hz

AMPLITUDE STATS
Min:           {min_amp:8.2f}
Max:           {max_amp:8.2f}
Mean:          {mean_amp:8.2f}

CURRENT PARAMETERS
Half-life:     {self.slider_hl.val:8.2f} s
Amp Thresh:    {self.slider_amp.val:8.2f}
SD Thresh:     {self.slider_sd.val:8.2f}

Note: Adjust sliders then
click 'Reprocess' to apply
"""
        self.stats_text.set_text(stats_str)

        # Set X-axis limits
        self.ax_gas.set_xlim(self.times[0], self.times[-1])
        self.ax_smoothed.set_xlim(self.times[0], self.times[-1])
        self.ax_derivative.set_xlim(self.times[0], self.times[-1])
        self.ax_second_derivative.set_xlim(self.times[0], self.times[-1])
        self.ax_bout_freq.set_xlim(self.times[0], self.times[-1])
        self.ax_amplitude_scatter.set_xlim(self.times[0], self.times[-1])

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# === MAIN ===
if __name__ == '__main__':
    import sys

    print("=" * 70)
    print(" OFFLINE BOUT DETECTION DASHBOARD")
    print("=" * 70)
    print()

    dashboard = OfflineDashboard(r"C:\Users\ltjth\Documents\Research\UKF_Data\Arr_Afr_run1.csv")

    print("\n" + "=" * 70)
    print(" INSTRUCTIONS")
    print("=" * 70)
    print("• Adjust sliders to change parameters")
    print("• Click 'Reprocess' button to recompute with new parameters")
    print("• GREEN segments = Accepted bouts")
    print("• RED segments = Rejected bouts")
    print("• Bout frequency uses 1-second sliding window")
    print("• Amplitude threshold affects acceptance (not reprocessing)")
    print("• Half-life and SD threshold require reprocessing")
    print("=" * 70 + "\n")

    dashboard.show()
