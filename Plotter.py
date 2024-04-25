import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import seaborn as sns

mpl.use('Agg')  # Ensure matplotlib does not use any Xwindows backend
mpl.rcParams.update({
    'figure.dpi' : 300,
    'axes.linewidth': 1,
    'axes.labelsize' : 15,
    'legend.fontsize': 12,
    'xtick.labelsize' : 11,
    'ytick.labelsize' : 13,
    'text.antialiased': True,
    'errorbar.capsize': 2
})

class Plotter:
    def __init__(self, filenames, title=None, prefix="output", cmap='magma', out_folder='.'):
        self.filenames = filenames
        self.title = title
        self.prefix = prefix
        self.out_folder = out_folder
        self.cmap = mpl.colormaps[cmap]  # Updated to use the new method

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    def open_file(self, filename):
        return h5py.File(filename, 'r')

    @property
    def num_solutions(self, fd):
        solutions = fd['Output']['Solutions']
        return sum('solution' in k for k in solutions)

    def solution_iter(self, fd):
        solutions = fd['Output']['Solutions']
        for k, v in solutions.items():
            if 'solution' in k:
                # Isolate and convert the numeric part of the key
                solution_number = ''.join(filter(str.isdigit, k))
                if solution_number.isdigit():
                    yield int(solution_number), v
                else:
                    print(f"Warning: Unable to parse solution index from key '{k}'")


    def open_file(self, filename):
        try:
            fd = h5py.File(filename, 'r')
            print(f"File opened successfully: {filename}")
            return fd
        except Exception as e:
            print(f"Failed to open file {filename}: {e}")
            return None

    @staticmethod
    def check_if_any_strings_exist(main_string, *substrings):
        return any(s in main_string for s in substrings)

    def create_grid_res(self, resolution, wave_min, wave_max):
        # Creating wavelength bins
        wave_list = [wave_min]
        wave = wave_min
        while wave < wave_max:
            delta_wave = wave / resolution
            wave += delta_wave
            wave_list.append(wave)
        return np.array(wave_list)

    def _generic_plot(self, ax, wlgrid, spectra, resolution=None, color=None, error=False, alpha=1.0, label=None):
        # Create a new grid at the specified resolution
        bin_edges = np.linspace(min(wlgrid) * 0.9, max(wlgrid) * 1.1, num=int((max(wlgrid) - min(wlgrid)) * resolution / max(wlgrid)) + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # print("wlgrid shape:", wlgrid.shape)
        # print("spectra['native_spectrum'] shape:", spectra['native_spectrum'].shape)

        # Check if wlgrid and spectra['native_spectrum'] align
        if wlgrid.shape != spectra['native_spectrum'].shape:
            raise ValueError("The wavelength grid and spectrum data do not align!")

        # Use numpy to bin the data
        binned_spectrum, _ = np.histogram(wlgrid, bins=bin_edges, weights=spectra['native_spectrum'])
        sum_weights, _ = np.histogram(wlgrid, bins=bin_edges, weights=np.ones_like(spectra['native_spectrum']))

        # Normalize the counts by the number of points in each bin to get the average
        binned_spectrum /= sum_weights
        where_are_NaNs = np.isnan(binned_spectrum)
        binned_spectrum[where_are_NaNs] = 0

        # Optional: Handle errors if provided and requested
        if 'native_std' in spectra:
            weights_spec = spectra['native_std'][...]
            sum_sq_errors, _ = np.histogram(wlgrid, bins=bin_edges, weights=weights_spec**2)
            binned_errors = np.sqrt(sum_sq_errors / sum_weights)
            binned_errors[where_are_NaNs] = 0

        # Plotting the binned or native spectrum
        ax.plot(bin_centers, binned_spectrum, label=label, alpha=alpha, color=color)

        # Plot error bars if available and requested
        #if error and binned_errors is not None:
        ax.fill_between(bin_centers, binned_spectrum - binned_errors, binned_spectrum + binned_errors, alpha=0.2, color=color, edgecolor='none')



################ Plotting ################
    def plot_temperature_profiles(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 10))
            own_figure = True
        else:
            own_figure = False

        for idx, filename in enumerate(self.filenames):
            fd = self.open_file(filename)
            if fd is not None:
                try:
                    for solution_index, solution_data in self.solution_iter(fd):
                        temperature = np.array(solution_data['Profiles/temp_profile'])
                        temperature_std = np.array(solution_data['Profiles/temp_profile_std'])
                        pressure = np.array(solution_data['Profiles/pressure_profile'])
                        if temperature is not None and pressure is not None:
                            ax.plot(temperature, pressure, label=f'{legend_tag[idx]}', color=self.cmap(float(idx)/len(self.filenames)))
                            ax.fill_betweenx(pressure, temperature-temperature_std, temperature+temperature_std, color=self.cmap(float(idx)/len(self.filenames)), alpha=0.2)
                finally:
                    fd.close()
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Pressure (Pa)')
        ax.set_ylim(1e2,1e6)
        ax.invert_yaxis()
        ax.set_yscale('log')
        

        if own_figure:
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{planet}_{mode}_temperature_profiles.pdf'))
            plt.close()

    def plot_fitted_spectrum(self, ax=None, resolution=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10.6, 7.0))
            own_figure = True
        else:
            own_figure = False

        for idx, filename in enumerate(self.filenames):
            fd = self.open_file(filename)
            if fd is not None:
                try:
                    obs_spectrum = fd['Observed']['spectrum'][...]
                    error = fd['Observed']['errorbars'][...]
                    wlgrid = fd['Observed']['wlgrid'][...]
                    ax.errorbar(wlgrid, obs_spectrum, yerr=error, fmt='o', color='black', alpha=0.4, label='Observed' if idx == 0 else "_nolegend_")

                    color = self.cmap(float(idx)/len(self.filenames))
                    for solution_index, solution_data in self.solution_iter(fd):
                        if resolution is None:
                            try:
                                binned_grid = solution_data['Spectra']['binned_wlgrid'][...]
                            except KeyError:
                                binned_grid = solution_data['Spectra']['bin_wlgrid'][...]
                            binned_spectrum = solution_data['Spectra']['binned_spectrum'][...]
                            binned_spectrum_std = solution_data['Spectra']['binned_std'][...]

                            ax.plot(binned_grid, binned_spectrum, color=color, label=f'{legend_tag[idx]}')
                            ax.fill_between(binned_grid, binned_spectrum - binned_spectrum_std, binned_spectrum + binned_spectrum_std, alpha=0.2, color=color, edgecolor='none')
                        else:
                            grid = solution_data['Spectra']['native_wlgrid'][...]
                            self._generic_plot(ax, grid, solution_data['Spectra'], resolution, color=color, label=f'{legend_tag[idx]}')   
                finally:
                    fd.close()
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Flux')
        ax.set_xlim(np.min(wlgrid)-0.05*np.min(wlgrid), np.max(wlgrid)+0.05*np.max(wlgrid))
        if np.max(wlgrid) - np.min(wlgrid) > 5:
            ax.set_xscale('log')
        ax.set_ylim(0.0252, 0.0256)
        ax.legend(loc='best',ncol=2,framealpha=0)

        # If the figure is owned by this function, finalize and show/save it
        if own_figure:
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{planet}_{mode}_superimposed_spectra.pdf'))
            plt.close()




    def plot_feature_distributions(self, ax=None):
        # Determine layout based on the provided ax
        if ax is None:
            fig, axes = plt.subplots(2, 5, figsize=(17, 6))  # Adjust the grid size based on the number of features
            axes = axes.flatten()  # Flatten to ease the iteration
            own_figure = True
        else:
            # Ensure ax is iterable and correctly formatted
            if not isinstance(ax, np.ndarray):
                axes = np.array([ax] * 10).flatten()  # Adjust this number based on expected number of plots
            else:
                axes = ax.flatten()
            own_figure = False

        cmap = plt.get_cmap(self.cmap)
        color_idx = 0

        for idx, filename in enumerate(self.filenames):
            fd = self.open_file(filename)
            if fd is not None:
                try:
                    for solution_index, solution_data in self.solution_iter(fd):
                        data = {param: solution_data['fit_params'][param]['trace'][...] for param in params if param in solution_data['fit_params']}
                        weights = solution_data['weights'][...]
                        

                        for axis, (param, values) in zip(axes, data.items()):
                            if mode == 'emission' and param == 'log_VO':# and 'log_TiO' not in solution_data['fit_params']:
                                # Ensure log_VO is always plotted on the last axis if log_TiO is missing
                                axis = axes[-1]
                                sns.kdeplot(x=values, weights=weights, ax=axis, color=cmap(color_idx / len(self.filenames)), fill=True)
                            elif mode == 'transmission' and param == 'T':
                                # Ensure T is always plotted on the last axis if transmission mode
                                axis = axes[-1]
                                sns.kdeplot(x=values, weights=weights, ax=axis, color=cmap(color_idx / len(self.filenames)), fill=True)
                            else:
                                sns.kdeplot(x=values, weights=weights, ax=axis, color=cmap(color_idx / len(self.filenames)), fill=True)
                            axis.set_title(f"{param}", fontsize=13)
                            axis.set_ylabel("")
                            axis.set_yticks([])

                        color_idx += 1
                finally:
                    fd.close()

        if own_figure:
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{planet}_{mode}_feature_distributions.pdf'))
            plt.close()



    def integrated_plot(self):
        # Create a figure with a gridspec layout
        fig = plt.figure(figsize=(14, 8), dpi=300)  # Adjust the figure size as needed
        fig.suptitle(f"{planet}", fontsize=20, y=0.95)
        gs = gridspec.GridSpec(2, 10, height_ratios=[5, 1])  # Two rows, with the first row being double height
        
        # Assign subplots to the gridspec positions
        main_spectrum_ax = fig.add_subplot(gs[0, 1:-3])  # Main spectrum plot spans the first four columns of the first row
        tp_profile_ax = fig.add_subplot(gs[0, -2:])  # Temperature-pressure profile in the last column of the first row

        # Create axes for feature distributions
        distribution_axes = [fig.add_subplot(gs[1, j]) for j in range(10)]
        
        

        # Call plotting functions and pass specific axes
        self.plot_fitted_spectrum(main_spectrum_ax, resolution=10000)  # Example resolution
        self.plot_temperature_profiles(tp_profile_ax)
        # for ax in distribution_axes_row1 + distribution_axes_row2:
        #     self.plot_feature_distributions(ax)  # Modify this method if it can't handle individual axes

        cmap = plt.get_cmap(self.cmap)
        color_idx = 0
        # Iterate over the files and parameters to plot each feature distribution
        for idx, filename in enumerate(self.filenames):
            fd = self.open_file(filename)
            if fd is not None:
                try:
                    color = self.cmap(idx / len(self.filenames))
                    for solution_index, solution_data in self.solution_iter(fd):
                        data = {param: solution_data['fit_params'][param]['trace'][...] for param in params if param in solution_data['fit_params']}
                        weights = solution_data['weights'][...]

                        for axis, (param, values) in zip(distribution_axes, data.items()):
                            if mode == 'emission' and param == 'log_VO':
                                # Ensure log_VO is always plotted on the last axis if log_TiO is missing
                                axis = distribution_axes[-1]
                                sns.kdeplot(x=values, weights=weights, ax=axis, color=cmap(color_idx / len(self.filenames)), fill=True)
                            if mode == 'transmission' and param == 'T':
                                # Ensure T is always plotted on the last axis if transmission mode
                                axis = distribution_axes[-1]
                                sns.kdeplot(x=values, weights=weights, ax=axis, color=cmap(color_idx / len(self.filenames)), fill=True)
                            else:
                                sns.kdeplot(x=values, weights=weights, ax=axis, color=cmap(color_idx / len(self.filenames)), fill=True)
                            axis.set_title(f"{param}", fontsize=13)
                            axis.set_ylabel("")
                            axis.set_yticks([])
                    color_idx += 1
                finally:
                    fd.close()
        
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{planet}_{mode}_integrated_plot.pdf'),bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    
    file_path = '/Users/deborah/Documents/Research/EXACT/results/exoplanets'

    # ################################# EMISSION ################################# 
    # mode = 'emission'

    # params = ['T_surface', 'T_point1', 'T_point2', 'T_point3', 'T_top', 'C_O_ratio', 'log_Kzz', 'log_metallicity', 'log_TiO', 'log_VO'] 
    # params = ['T_surface', 'T_point1', 'T_point2', 'T_point3', 'T_top', 'log_C_O_ratio', 'log_Kzz', 'log_metallicity', 'log_TiO', 'log_VO'] 

    #vlegend_tag = ['FRECKLL', 'FRECKLL, TiO','FRECKLL, VO','FRECKLL, TiO & VO']
    # legend_tag = ['FRECKLL', 'FRECKLL, log(Z)>0', 'FRECKLL, TiO, log(Z)>0','FRECKLL, VO, log(Z)>0','FRECKLL, TiO & VO, log(Z)>0',
    #              'FRECKLL, log(Z)<0', 'FRECKLL, TiO, log(Z)<0','FRECKLL, VO, log(Z)<0','FRECKLL, TiO & VO, log(Z)<0']

    # planet = 'HAT-P-2 b'
    # plotter = Plotter(filenames=[f"{file_path}/HATP2b/HATP2b_retrieval_pychegp.hdf5",
    #                              f"{file_path}/HATP2b/HATP2b_retrieval_pychegp_logZpos.hdf5", 
    #                              f"{file_path}/HATP2b/HATP2b_retrieval_pychegp_logZpos_freeTiO.hdf5",
    #                              f"{file_path}/HATP2b/HATP2b_retrieval_pychegp_logZpos_freeVO.hdf5",
    #                              f"{file_path}/HATP2b/HATP2b_retrieval_pychegp_logZpos_freeTiOVO.hdf5",
    #                              f"{file_path}/HATP2b/HATP2b_retrieval_pychegp_logZneg.hdf5", 
    #                              f"{file_path}/HATP2b/HATP2b_retrieval_pychegp_logZneg_freeTiO.hdf5",
    #                              f"{file_path}/HATP2b/HATP2b_retrieval_pychegp_logZneg_freeVO.hdf5",
    #                              f"{file_path}/HATP2b/HATP2b_retrieval_pychegp_logZneg_freeTiOVO.hdf5"
    #                              ])

    # planet = 'HD-189733 b'
    # plotter = Plotter(filenames=[f"{file_path}/HD189733b/HD189733b_retrieval_pychegp.hdf5",
    #                              f"{file_path}/HD189733b/HD189733b_retrieval_pychegp_logZpos.hdf5", 
    #                              f"{file_path}/HD189733b/HD189733b_retrieval_pychegp_logZpos_freeTiO.hdf5",
    #                              f"{file_path}/HD189733b/HD189733b_retrieval_pychegp_logZpos_freeVO.hdf5",
    #                              f"{file_path}/HD189733b/HD189733b_retrieval_pychegp_logZpos_freeTiOVO.hdf5",
    #                              f"{file_path}/HD189733b/HD189733b_retrieval_pychegp_logZneg.hdf5", 
    #                              f"{file_path}/HD189733b/HD189733b_retrieval_pychegp_logZneg_freeTiO.hdf5",
    #                              f"{file_path}/HD189733b/HD189733b_retrieval_pychegp_logZneg_freeVO.hdf5",
    #                              f"{file_path}/HD189733b/HD189733b_retrieval_pychegp_logZneg_freeTiOVO.hdf5"
    #                              ])

    # planet = 'TrES-3 b'
    # plotter = Plotter(filenames=[f"{file_path}/TrES3b/TrES3b_retrieval_pychegp_photodissociation.hdf5", 
    #                              f"{file_path}/TrES3b/TrES3b_retrieval_pychegp_freeTiO.hdf5",
    #                              f"{file_path}/TrES3b/TrES3b_retrieval_pychegp_freeVO.hdf5",
    #                              f"{file_path}/TrES3b/TrES3b_retrieval_pychegp_freeTiOVO.hdf5"])
    
    # planet = 'WASP-19 b'
    # plotter = Plotter(filenames=[f"{file_path}/WASP19b/WASP19b_retrieval_pychegp_photodissociation.hdf5",
    #                              f"{file_path}/WASP19b/WASP19b_retrieval_pychegp_photodissociation_logZpos.hdf5", 
    #                              f"{file_path}/WASP19b/WASP19b_retrieval_pychegp_logZpos_freeTiO.hdf5",
    #                              f"{file_path}/WASP19b/WASP19b_retrieval_pychegp_logZpos_freeVO.hdf5",
    #                              f"{file_path}/WASP19b/WASP19b_retrieval_pychegp_logZpos_freeTiOVO.hdf5",
    #                              f"{file_path}/WASP19b/WASP19b_retrieval_pychegp_photodissociation_logZneg.hdf5", 
    #                              f"{file_path}/WASP19b/WASP19b_retrieval_pychegp_logZneg_freeTiO.hdf5",
    #                              f"{file_path}/WASP19b/WASP19b_retrieval_pychegp_logZneg_freeVO.hdf5",
    #                              f"{file_path}/WASP19b/WASP19b_retrieval_pychegp_logZneg_freeTiOVO.hdf5"
    #                              ])
    
    # planet = 'WASP-4 b'
    # plotter = Plotter(filenames=[f"{file_path}/WASP4b/WASP4b_retrieval_pychegp.hdf5", 
    #                              f"{file_path}/WASP4b/WASP4b_retrieval_pychegp_freeTiO.hdf5",
    #                              f"{file_path}/WASP4b/WASP4b_retrieval_pychegp_freeVO.hdf5",
    #                              f"{file_path}/WASP4b/WASP4b_retrieval_pychegp_freeTiOVO.hdf5"])
    
    # planet = 'WASP-43 b'
    # plotter = Plotter(filenames=[f"{file_path}/WASP43b/WASP43b_retrieval_pychegp.hdf5", 
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_freeTiO.hdf5",
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_freeVO.hdf5",
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_freeTiOVO.hdf5"])
    
    # planet = 'WASP-77 A b'
    # plotter = Plotter(filenames=[f"{file_path}/WASP77Ab/WASP77Ab_retrieval_pychegp.hdf5", 
    #                              f"{file_path}/WASP77Ab/WASP77Ab_retrieval_pychegp_freeTiO_photodissociation.hdf5",
    #                              f"{file_path}/WASP77Ab/WASP77Ab_retrieval_pychegp_freeVO_photodissociation.hdf5",
    #                              f"{file_path}/WASP77Ab/WASP77Ab_retrieval_pychegp_freeTiOVO_photodissociation.hdf5"])
    






    ################################# TRANSMISSION ################################# 
    mode = 'transmission'
    params = ['planet_radius','C_O_ratio', 'log_Kzz', 'log_metallicity', 'T_surface', 'T_point1', 'T_point2', 'T_point3','T_top','T']

    legend_tag = ['FRECKLL, 5-point temperature profile', 'FRECKLL, isothermal temperature profile']
    
    planet = 'WASP-43 b'
    plotter = Plotter(filenames=[f"{file_path}/WASP43b/WASP43b_retrieval_transmission_pychegp.hdf5", 
                                 f"{file_path}/WASP43b/WASP43b_retrieval_transmission_pychegp_isothermal.hdf5"])


    plotter.plot_temperature_profiles()
    plotter.plot_fitted_spectrum(resolution=10000)
    #plotter.plot_fitted_spectrum()
    plotter.plot_feature_distributions()
    plotter.integrated_plot()
    