import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Constants
w = 20e-3
h = 50e-3
t = 0.1e-3
a_block = 2e-3
mwing = 0.85e-3
msupport = 9.92e-6
mrod = msupport
mtotal = mwing+msupport+mrod
r = 0.5e-3
E = 0.8e6
G = E / (2*(1 + 0.48))

# Moment of inertia
Iwing = (mwing/48)*((h**2 + w**2) + (h**2 + t**2) +
                    (w**2*np.cos(np.pi/4)**2 + t**2*np.sin(np.pi/4) + h**2) +
                    (w**2*np.cos(3*np.pi/4)**2 + t**2*np.sin(3*np.pi/4) + h**2))
Isupport = msupport * (2 * a_block**2) / 12
Irod = (mrod * r**2) / 2
Itotal = Iwing + Isupport + Irod

#print(Itotal)

# Create GUI
root = tk.Tk()
root.title("Torsional Spring & Resonance Explorer")

# Labels
k_label = ttk.Label(root, text="k = ")
k_label.grid(row=3, column=0, columnspan=2, pady=10)
steel_label = ttk.Label(root, text="Fr (Steel) = ")
steel_label.grid(row=4, column=0, columnspan=2)
sil_label = ttk.Label(root, text="Fr (Silicone) = ")
sil_label.grid(row=5, column=0, columnspan=2)



# Update function
def update_vals(*args):
    b_mm = float(b_scale.get())
    H_mm = float(H_scale.get())

    b = b_mm / 1000
    H = H_mm / 1000

    r_hole = 2.9E-3    #0.75e-3  # 1.5 mm diameter / 2

    J = (1 / 3) * b * H ** 3 #- (np.pi * r_hole ** 4) / 2

    k = (G * J) / H
    #print(np.sqrt(k / mtotal))
    Fr_steel = (1 /2)*np.pi * np.sqrt((1.2E-3)/Itotal)

    Fr_sil = (1 / 2)*np.pi * np.sqrt((k/Itotal))


    k_label.config(text=f"Silicone k = {k:.8f} N·m/rad   Steel k = 0.0012 Nm/rad")
    steel_label.config(text=f" resonant freq (Steel) = {Fr_steel:.2f} Hz")
    sil_label.config(text=f"resonant freq (Silicone) = {Fr_sil:.2f} Hz")


import numpy as np


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

def plot_frequency_heatmap():
    # Densities (kg/m^3)
    density_wing = 1200
    density_support = 7800
    density_rod = 7800

    # Fixed wing dimensions except height
    w_fixed = 20e-3
    t_fixed = 0.1e-3

    # Rod dimensions fixed
    rod_radius = 0.5e-3  # 1mm diameter / 2
    rod_length = 5e-3

    # Support fixed cube
    a_block = 2e-3

    E = 0.8e6
    G = E / (2 * (1 + 0.48))

    # Sweep ranges (meters)
    b_range = np.linspace(2e-3, 10e-3, 30)
    H_range = np.linspace(1.5e-3, 5e-3, 30)
    #h_range = np.linspace(10e-3, 50e-3, 30)

    results = []

    for b in b_range:
        for H in H_range:
            #for h in h_range:

            volume_wing = w_fixed * h * t_fixed
            volume_support = a_block ** 3
            volume_rod = np.pi * rod_radius ** 2 * rod_length

            mass_wing = density_wing * volume_wing
            mass_support = density_support * volume_support
            mass_rod = density_rod * volume_rod

            Iwing = (mass_wing / 48) * (
                    (h ** 2 + w_fixed ** 2) +
                    (h ** 2 + t_fixed ** 2) +
                    (w_fixed ** 2 * np.cos(np.pi / 4) ** 2 + t_fixed ** 2 * np.sin(np.pi / 4) + h ** 2) +
                    (w_fixed ** 2 * np.cos(3 * np.pi / 4) ** 2 + t_fixed ** 2 * np.sin(3 * np.pi / 4) + h ** 2)
            )
            Isupport = mass_support * (a_block ** 2) / 6
            Irod = (mass_rod * rod_radius ** 2) / 2

            Itotal = Iwing + Isupport + Irod

            r_hole = 0.75e-3  # 1.5 mm diameter / 2

            J = (1 / 3) * b * H ** 3 #- (np.pi * r_hole ** 4) / 2


            k = (G * J) / H

            freq = (1 / (2 * np.pi)) * np.sqrt(k / Itotal)

            if 48 <= freq <= 50:
                results.append({
                    'Spring Width b (mm)': b * 1e3,
                    'Spring Thickness H (mm)': H * 1e3,
                    'Wing Height h (mm)': h * 1e3,
                    'Frequency (Hz)': freq
                })

    if not results:
        print("No results found in the 40–50 Hz range.")
        return

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv("sweep_results_50height.csv", index=False)

    # Extract data for plotting
    b_vals = df['Spring Width b (mm)']
    H_vals = df['Spring Thickness H (mm)']
    h_vals = df['Wing Height h (mm)']

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    sc = plt.scatter(b_vals, H_vals, c=h_vals, cmap='Reds', alpha=1.0)

    plt.xlabel('Spring Width b (mm)')
    plt.ylabel('Spring Thickness H (mm)')
    plt.title('Parameter Sweep of Silicone Suspension')

    cbar = plt.colorbar(sc)
    cbar.set_label('Wing Height h (mm)')

    plt.tight_layout()
    plt.show()

    return df



# b slider
ttk.Label(root, text="Width (mm)").grid(row=0, column=0, padx=10, sticky="w")
b_scale = tk.Scale(root, from_=0.1, to=10, resolution=0.1,
                   orient="horizontal", length=300, command=update_vals)
b_scale.set(2.0)
b_scale.grid(row=0, column=1)

# H slider
ttk.Label(root, text="Thickness (mm)").grid(row=1, column=0, padx=10, sticky="w")
H_scale = tk.Scale(root, from_=0.1, to=10, resolution=0.1,
                   orient="horizontal", length=300, command=update_vals)
H_scale.set(0.1)
H_scale.grid(row=1, column=1)

# Initial update
update_vals()
#plot_frequency_heatmap()


root.mainloop()
