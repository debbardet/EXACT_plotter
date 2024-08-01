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

def decode_string_array(f):
        """Helper to decode strings from hdf5"""
        sl = list(f)
        return [s[0].decode('utf-8') for s in sl] 

class Plotter:
    def __init__(self, filenames, title=None, prefix="output", cmap='magma', out_folder='plots/'):
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
    
    #@property
    def activeGases(self, fd):
        return decode_string_array(fd['ModelParameters']['Chemistry']['active_gases'])

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
            plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{planet}_{mode}_temperature_profiles.pdf'.replace(' ', '_')))
            plt.close()

    def plot_chemistry_profiles(self, ax=None):
        chem_cmap = mpl.colormaps['jet']
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 10))
            own_figure = True
        
            for idx, filename in enumerate(self.filenames):
                fd = self.open_file(filename)
                if fd is not None:
                    try:
                        for solution_index, solution_data in self.solution_iter(fd):
                            active_profile = np.array(solution_data['Profiles/active_mix_profile'])
                            active_profile_std = np.array(solution_data['Profiles/active_mix_profile_std'])
                            pressure = np.array(solution_data['Profiles/pressure_profile'])

                            num_moles = len(self.activeGases(fd))
                            cols_mol = {}
                            for mol_idx,mol_name in enumerate(self.activeGases(fd)):
                                cols_mol[mol_name] = chem_cmap(mol_idx/num_moles)

                                prof = active_profile[mol_idx]
                                prof_std = active_profile_std[mol_idx]

                                ax.plot(prof,pressure,color=cols_mol[mol_name], label=mol_name)

                                # ax.fill_betweenx(pressure, prof + prof_std, prof,
                                #                 color=chem_cmap(mol_idx / num_moles), alpha=0.5)
                                # ax.fill_betweenx(pressure, prof,
                                #                 np.power(10, (np.log10(prof) - (
                                #                             np.log10(prof + prof_std) - np.log10(prof)))),
                                #                 color=chem_cmap(mol_idx / num_moles), alpha=0.5)
                    finally:
                        fd.close()

        else:
            own_figure = False

            if mode =='emission' and planet == 'WASP-77 A b' or planet =='WASP-74 b' and mode =='emission':
                filename = self.filenames[1] 
            elif mode =='emission' and planet == 'HAT-P-2 b' or mode =='emission' and planet =='HD-189733 b' or mode =='emission' and planet =='WASP-19 b':
                filename = self.filenames[5]
            else:
                filename = self.filenames[0]
            fd = self.open_file(filename)
            if fd is not None:
                try:
                    for solution_index, solution_data in self.solution_iter(fd):
                        active_profile = np.array(solution_data['Profiles/active_mix_profile'])
                        active_profile_std = np.array(solution_data['Profiles/active_mix_profile_std'])
                        pressure = np.array(solution_data['Profiles/pressure_profile'])

                        num_moles = len(self.activeGases(fd))
                        cols_mol = {}
                        for mol_idx,mol_name in enumerate(self.activeGases(fd)):
                            cols_mol[mol_name] = chem_cmap(mol_idx/num_moles)

                            prof = active_profile[mol_idx]
                            prof_std = active_profile_std[mol_idx]

                            ax.plot(prof,pressure,color=cols_mol[mol_name], label=mol_name)

                            # ax.fill_betweenx(pressure, prof + prof_std, prof,
                            #                 color=chem_cmap(mol_idx / num_moles), alpha=0.5)
                            # ax.fill_betweenx(pressure, prof,
                            #                 np.power(10, (np.log10(prof) - (
                            #                             np.log10(prof + prof_std) - np.log10(prof)))),
                            #                 color=chem_cmap(mol_idx / num_moles), alpha=0.5)
                finally:
                    fd.close()

   
        ax.set_xlabel('Volume mixing ratio')
        ax.set_xscale('log')
        ax.set_ylabel('Pressure (Pa)')
        ax.set_ylim(1e2,1e6)
        ax.invert_yaxis()
        ax.set_yscale('log')
        ax.legend()
        

        if own_figure:
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{planet}_{mode}_chemistry_profiles.pdf'.replace(' ', '_')))
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
        ax.set_ylim(0.0, 0.0010)
        ax.legend(loc='lower left',ncol=2,framealpha=0,prop={'size': 10})

        # If the figure is owned by this function, finalize and show/save it
        if own_figure:
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{planet}_{mode}_superimposed_spectra.pdf'.replace(' ', '_')))
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
                            if mode == 'emission' and param == 'log_VO' and axis == axes[-2]:
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
            plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{planet}_{mode}_feature_distributions.pdf'.replace(' ', '_')))
            plt.close()



    def integrated_plot(self):
        # Create a figure with a gridspec layout
        fig = plt.figure(figsize=(16, 7), dpi=300)  # Adjust the figure size as needed
        fig.suptitle(f"{planet}", fontsize=20, y=0.95)
        gs = gridspec.GridSpec(2, 20, height_ratios=[4, 1])  # Two rows, with the first row being double height
        
        # Assign subplots to the gridspec positions
        main_spectrum_ax = fig.add_subplot(gs[0, 1:9])  # Main spectrum plot spans the first four columns of the first row
        tp_profile_ax = fig.add_subplot(gs[0, 10:-6])    # Temperature-pressure profile in the middle column of the first row
        chem_profile_ax = fig.add_subplot(gs[0, -5:])   # Chemical-pressure profiles in the last column of the first row

        # Create axes for feature distributions
        distribution_axes = [fig.add_subplot(gs[1, j:j+2]) for j in range(0,20,2)]
        
        

        # Call plotting functions and pass specific axes
        self.plot_fitted_spectrum(main_spectrum_ax, resolution=10000)  # Example resolution
        self.plot_temperature_profiles(tp_profile_ax)
        self.plot_chemistry_profiles(chem_profile_ax)
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
 
                        iparam = 0
                        for axis, (param, values) in zip(distribution_axes, data.items()):
                            if mode == 'emission' and param == 'log_VO':
                                # Ensure log_VO is always plotted on the last axis if log_TiO is missing
                                axis = distribution_axes[-1]
                                iparam =-1
                                sns.kdeplot(x=values, weights=weights, ax=axis, color=cmap(color_idx / len(self.filenames)), fill=True)
                            if mode == 'transmission' and param == 'T':
                                # Ensure T is always plotted on the last axis if transmission mode
                                axis = distribution_axes[-1]
                                iparam =-1
                                sns.kdeplot(x=values, weights=weights, ax=axis, color=cmap(color_idx / len(self.filenames)), fill=True)
                            else:
                                sns.kdeplot(x=values, weights=weights, ax=axis, color=cmap(color_idx / len(self.filenames)), fill=True)
                            axis.set_title(param_title[iparam], fontsize=13)
                            axis.tick_params(axis='x', labelrotation=45)
                            axis.set_ylabel("")
                            axis.set_yticks([])
                            iparam += 1
                    color_idx += 1
                finally:
                    fd.close()
        
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{planet}_{mode}_integrated_plot.pdf'.replace(' ', '_')),bbox_inches='tight')
        plt.close()



    def multi_spectrum_plot(self, resolution):
        # Create a figure 
        fig, ax = plt.subplots(int(len(self.filenames)), figsize=(7,16), dpi=300)  # Adjust the figure size as needed
                
        for idx, filename in enumerate(self.filenames):
            fd = self.open_file(filename)
            if fd is not None:
                try:
                    obs_spectrum = fd['Observed']['spectrum'][...]
                    error = fd['Observed']['errorbars'][...]
                    wlgrid = fd['Observed']['wlgrid'][...]
                    ax[idx].errorbar(wlgrid, obs_spectrum, yerr=error, fmt='o', color='black', alpha=1, label='Observed' if idx == 0 else "_nolegend_")
                    ax[idx].legend(bbox_to_anchor=(0.7, 1.15), loc='best',framealpha=0) if idx == 0 else "_nolegend_"

                    color = 'crimson'
                    for solution_index, solution_data in self.solution_iter(fd):
                        if resolution is None:
                            try:
                                binned_grid = solution_data['Spectra']['binned_wlgrid'][...]
                            except KeyError:
                                binned_grid = solution_data['Spectra']['bin_wlgrid'][...]
                            binned_spectrum = solution_data['Spectra']['binned_spectrum'][...]
                            binned_spectrum_std = solution_data['Spectra']['binned_std'][...]

                            ax[idx].plot(binned_grid, binned_spectrum, color=color)
                            ax[idx].fill_between(binned_grid, binned_spectrum - binned_spectrum_std, binned_spectrum + binned_spectrum_std, alpha=0.2, color=color, edgecolor='none')
                        else:
                            grid = solution_data['Spectra']['native_wlgrid'][...]
                            self._generic_plot(ax[idx], grid, solution_data['Spectra'], resolution, color=color)
                               
                finally:
                    fd.close()
                
            ax[idx].set_xlim(np.min(wlgrid)-0.05*np.min(wlgrid), np.max(wlgrid)+0.05*np.max(wlgrid))
            if np.max(wlgrid) - np.min(wlgrid) > 5:
                ax[idx].set_xscale('log')
            ax[idx].set_ylim(np.min(obs_spectrum-error-0.1*error), np.max(obs_spectrum+error+0.1*error))
            ax[idx].set_title(f'{legend_tag[idx]}')
            
            
        fig.supxlabel('Wavelength (μm)',fontsize=18)
        fig.supylabel('Flux',fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{mode}_superimposed_spectra.pdf'.replace(' ', '_')))
        plt.close()




    def CO_Z_plot(self):
        # Create a figure 
        fig = plt.figure(figsize=(6,4), dpi=300)
        ax = fig.add_subplot(1,1,1)
        for idx, filename in enumerate(self.filenames):
            fd = self.open_file(filename)
            if fd is not None:
                try:
                    color = self.cmap(float(idx)/len(self.filenames)) 
                    C_O_ratio = fd['Output']['Solutions']['solution0']['fit_params']['C_O_ratio']['value'][...]
                    C_O_ratioerror = fd['Output']['Solutions']['solution0']['fit_params']['C_O_ratio']['sigma_p'][...]
                    log_metallicity = fd['Output']['Solutions']['solution0']['fit_params']['log_metallicity']['value'][...]
                    log_metallicityerror = fd['Output']['Solutions']['solution0']['fit_params']['log_metallicity']['sigma_p'][...]
                    ax.errorbar(log_metallicity, C_O_ratio, xerr=log_metallicityerror, yerr=C_O_ratioerror, fmt='o', color=color, alpha=1, label=f'{legend_tag[idx]}')
                    if idx >= len(self.filenames)-2 : #legend_tag[idx]== 'WASP-77 A b':
                        ax.annotate(f'{legend_tag[idx]}', (log_metallicity, C_O_ratio), textcoords='offset points', xytext=(-60,-6), ha='left', va='top', color=color)
                    else:
                        ax.annotate(f'{legend_tag[idx]}', (log_metallicity, C_O_ratio), textcoords='offset points', xytext=(3,2), ha='left', va='bottom', color=color)

                finally:
                    fd.close()
        # ax.legend(loc='best',ncol=3,framealpha=0)                
        plt.xlim(-2.1,3)
        plt.ylim(0, 1.2)
        plt.xlabel('Retrieved Metallicity (O/H) [log]',fontsize=18)
        plt.ylabel('Retrieved C/O ratio',fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{mode}_CO_Z.pdf'.replace(' ', '_')))
        plt.close()



    def Teq_Z_plot(self):
        # Create a figure 
        fig = plt.figure(figsize=(6,4), dpi=300)
        ax = fig.add_subplot(1,1,1)
        for idx, filename in enumerate(self.filenames):
            fd = self.open_file(filename)
            if fd is not None:
                try:
                    color = self.cmap(float(idx)/len(self.filenames)) 
                    if mode == 'population_transmission':
                        Tequi = fd['ModelParameters']['Temperature']['T'][...]
                    else:
                        Tequi = fd['ModelParameters']['Temperature']['T_surface'][...]
                    # Tequierror = fd['ModelParameters']['Temperature']['T_surface'][...]
                    log_metallicity = fd['Output']['Solutions']['solution0']['fit_params']['log_metallicity']['value'][...]
                    log_metallicityerror = fd['Output']['Solutions']['solution0']['fit_params']['log_metallicity']['sigma_p'][...]
                    ax.errorbar(log_metallicity, Tequi, xerr=log_metallicityerror, fmt='o', color=color, alpha=1, label=f'{legend_tag[idx]}')
                    if idx >= len(self.filenames)-2 : #legend_tag[idx]== 'WASP-77 A b':
                        ax.annotate(f'{legend_tag[idx]}', (log_metallicity, Tequi), textcoords='offset points', xytext=(-60,-6), ha='left', va='top', color=color)
                    else:
                        ax.annotate(f'{legend_tag[idx]}', (log_metallicity, Tequi), textcoords='offset points', xytext=(3,2), ha='left', va='bottom', color=color)

                finally:
                    fd.close()
        # ax.legend(loc='best',ncol=3,framealpha=0)                
        plt.xlim(-2.1,3)
        plt.ylim(1100, 3400)
        plt.xlabel('Retrieved Metallicity (O/H) [log]',fontsize=18)
        plt.ylabel('Equilibrium Temperature [K]',fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{mode}_Tequi_Z.pdf'.replace(' ', '_')))
        plt.close()

    def Teq_CO_plot(self):
        # Create a figure 
        fig = plt.figure(figsize=(6,4), dpi=300)
        ax = fig.add_subplot(1,1,1)
        for idx, filename in enumerate(self.filenames):
            fd = self.open_file(filename)
            if fd is not None:
                try:
                    color = self.cmap(float(idx)/len(self.filenames)) 
                    C_O_ratio = fd['Output']['Solutions']['solution0']['fit_params']['C_O_ratio']['value'][...]
                    C_O_ratioerror = fd['Output']['Solutions']['solution0']['fit_params']['C_O_ratio']['sigma_p'][...]
                    if mode == 'population_transmission':
                        Tequi = fd['ModelParameters']['Temperature']['T'][...]
                    else:
                        Tequi = fd['ModelParameters']['Temperature']['T_surface'][...]
                    # Tequierror = fd['ModelParameters']['Temperature']['T_surface'][...]
                    ax.errorbar(C_O_ratio, Tequi, xerr=C_O_ratioerror, fmt='o', color=color, alpha=1, label=f'{legend_tag[idx]}')
                    if idx >= len(self.filenames)-2 : #legend_tag[idx]== 'WASP-77 A b':
                        ax.annotate(f'{legend_tag[idx]}', (C_O_ratio, Tequi), textcoords='offset points', xytext=(-60,-6), ha='left', va='top', color=color)
                    else:
                        ax.annotate(f'{legend_tag[idx]}', (C_O_ratio, Tequi), textcoords='offset points', xytext=(3,2), ha='left', va='bottom', color=color)

                finally:
                    fd.close()
        # ax.legend(loc='best',ncol=3,framealpha=0)                
        plt.ylim(1100, 3400)
        plt.xlim(0, 1.2)
        plt.xlabel('Retrieved C/O ratio',fontsize=18)
        plt.ylabel('Equilibrium Temperature [K]',fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{mode}_Tequi_CO.pdf'.replace(' ', '_')))
        plt.close()


    def CO_ZpZstar_plot(self):
        # Create a figure 
        fig = plt.figure(figsize=(6,4), dpi=300)
        ax = fig.add_subplot(1,1,1)
        for idx, filename in enumerate(self.filenames):
            fd = self.open_file(filename)
            if fd is not None:
                try:
                    color = self.cmap(float(idx)/len(self.filenames)) 
                    C_O_ratio = fd['Output']['Solutions']['solution0']['fit_params']['C_O_ratio']['value'][...]
                    C_O_ratioerror = fd['Output']['Solutions']['solution0']['fit_params']['C_O_ratio']['sigma_p'][...]
                    log_metallicity = fd['Output']['Solutions']['solution0']['fit_params']['log_metallicity']['value'][...]
                    log_metallicityerror = fd['Output']['Solutions']['solution0']['fit_params']['log_metallicity']['sigma_p'][...]
                    metallicityStar = fd['ModelParameters']['Star']['metallicity'][...]
                    metallicityStar = np.log10(metallicityStar)
                    ax.errorbar(C_O_ratio, log_metallicity/metallicityStar, yerr=log_metallicityerror, xerr=C_O_ratioerror, fmt='o', color=color, alpha=1, label=f'{legend_tag[idx]}')
                    if idx >= len(self.filenames)-2 : #legend_tag[idx]== 'WASP-77 A b':
                        ax.annotate(f'{legend_tag[idx]}', (C_O_ratio, log_metallicity/metallicityStar), textcoords='offset points', xytext=(3,-6), ha='left', va='top', color=color)
                    else:
                        ax.annotate(f'{legend_tag[idx]}', (C_O_ratio, log_metallicity/metallicityStar), textcoords='offset points', xytext=(3,2), ha='left', va='bottom', color=color)

                finally:
                    fd.close()
        plt.xlabel('Retrieved C/O ratio',fontsize=18)
        plt.ylabel(r'log$_{10}(Z_{planet}/Z_{star})$',fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{mode}_CO_ZpZstar.pdf'.replace(' ', '_')))
        plt.close()




    def LogFree_Temp_plot(self):
        # List of free species added to FRECKLL
        params = ['log_TiO', 'log_VO']
        for param in params:
            # Create a figure 
            fig = plt.figure(figsize=(6,4), dpi=300)
            ax = fig.add_subplot(1,1,1)
            for idx, filename in enumerate(self.filenames):
                fd = self.open_file(filename)
                if fd is not None:
                    try:
                        color = self.cmap(float(idx)/len(self.filenames)) 
                        LogFree = fd['Output']['Solutions']['solution0']['fit_params'][param]['value'][...]
                        LogFreeerror = fd['Output']['Solutions']['solution0']['fit_params'][param]['sigma_p'][...]
                        T = fd['Output']['Solutions']['solution0']['fit_params']['T_point1']['value'][...]
                        Terror = fd['Output']['Solutions']['solution0']['fit_params']['T_point1']['sigma_p'][...]
                        ax.errorbar(T, LogFree, xerr=Terror, yerr=LogFreeerror, fmt='o', color=color, alpha=1, label=f'{legend_tag[idx]}')
                        if idx >= len(self.filenames)-2 : #legend_tag[idx]== 'WASP-77 A b':
                            ax.annotate(f'{legend_tag[idx]}', (T, LogFree), textcoords='offset points', xytext=(3,-6), ha='left', va='top', color=color)
                        else:
                            ax.annotate(f'{legend_tag[idx]}', (T, LogFree), textcoords='offset points', xytext=(3,2), ha='left', va='bottom', color=color)

                    finally:
                        fd.close()
            plt.xlabel('Retrieved Planet Temperature [K]',fontsize=12)
            plt.ylabel('Log(TiO)',fontsize=12) if param == 'log_TiO' else plt.ylabel('Log(VO)',fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{mode}_{param}_temperature.pdf'.replace(' ', '_')))
            plt.close()



    def IsoT_Rp_plot(self):
        # Create a figure 
        fig = plt.figure(figsize=(6,4), dpi=300)
        ax = fig.add_subplot(1,1,1)
        for idx, filename in enumerate(self.filenames):
            fd = self.open_file(filename)
            if fd is not None:
                try:
                    color = self.cmap(float(idx)/len(self.filenames)) 
                    planet_radius = fd['Output']['Solutions']['solution0']['fit_params']['planet_radius']['value'][...]
                    planet_radiuserror = fd['Output']['Solutions']['solution0']['fit_params']['planet_radius']['sigma_p'][...]
                    T = fd['Output']['Solutions']['solution0']['fit_params']['T']['value'][...]
                    Terror = fd['Output']['Solutions']['solution0']['fit_params']['T']['sigma_p'][...]
                    ax.errorbar(planet_radius, T,  yerr=Terror, xerr=planet_radiuserror, fmt='o', color=color, alpha=1, label=f'{legend_tag[idx]}')
                    if idx >= len(self.filenames)-2 : #legend_tag[idx]== 'WASP-77 A b':
                        ax.annotate(f'{legend_tag[idx]}', (planet_radius, T), textcoords='offset points', xytext=(3,-6), ha='left', va='top', color=color)
                    else:
                        ax.annotate(f'{legend_tag[idx]}', (planet_radius, T), textcoords='offset points', xytext=(3,2), ha='left', va='bottom', color=color)

                finally:
                    fd.close()
        plt.xlabel(r'Retrieved Planet Radius [R$_{J}$]',fontsize=12)
        plt.ylabel('Retrieved Planet Temperature [K]',fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_folder, f'{self.prefix}_{mode}_IsoT_Rp.pdf'.replace(' ', '_')))
        plt.close()

if __name__ == "__main__":
    
    file_path = '/Users/deborah/Documents/Research/EXACT/results/exoplanets'

    # ################################# EMISSION ################################# 
    mode = 'emission'

    params = ['T_surface', 'T_point1', 'T_point2', 'T_point3', 'T_top', 'C_O_ratio', 'log_Kzz', 'log_metallicity', 'log_TiO', 'log_VO']
    param_title = [r'T$_{surface}$', r'T$_{1}$',r'T$_{2}$', r'T$_{3}$', r'T$_{top}$', r'C/O', 'log(Kzz)', 'log(Z)', 'log(TiO)', 'log(VO)'] 

    legend_tag = ['FRECKLL', 'FRECKLL, TiO','FRECKLL, VO','FRECKLL, TiO & VO']
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

    # planet = 'HD-209458 b'
    # plotter = Plotter(filenames=[f"{file_path}/HD209458b/HD209458b_retrieval_pychegp.hdf5", 
    #                              f"{file_path}/HD209458b/HD209458b_retrieval_pychegp_freeTiO.hdf5",
    #                              f"{file_path}/HD209458b/HD209458b_retrieval_pychegp_freeVO.hdf5",
    #                              f"{file_path}/HD209458b/HD209458b_retrieval_pychegp_freeTiOVO.hdf5"])
    
    # planet = 'Kepler-13 A b'
    # plotter = Plotter(filenames=[f"{file_path}/Kepler13Ab/Kepler13Ab_retrieval_pychegp.hdf5", 
    #                              f"{file_path}/Kepler13Ab/Kepler13Ab_retrieval_pychegp_freeTiO.hdf5",
    #                              f"{file_path}/Kepler13Ab/Kepler13Ab_retrieval_pychegp_freeVO.hdf5",
    #                              f"{file_path}/Kepler13Ab/Kepler13Ab_retrieval_pychegp_freeTiOVO.hdf5"
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
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_freeTiOVO.hdf5"
    #                             ])

    # planet = 'WASP-43 b log(Z)>0'
    # plotter = Plotter(filenames=[f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_logZpos.hdf5", 
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_logZpos_freeTiO.hdf5",
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_logZpos_freeVO.hdf5",
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_logZpos_freeTiOVO.hdf5"
    #                             ])
    
    planet = 'WASP-43 b     (Kreidberg+2014)'
    plotter = Plotter(filenames=[f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_Kreidberg_reduction.hdf5", 
                                 f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_freeTiO_Kreidberg_reduction.hdf5",
                                 f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_freeVO_Kreidberg_reduction.hdf5",
                                 f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_freeTiOVO_Kreidberg_reduction.hdf5"
                                ])

    # planet = 'WASP-43 b     log(Z)>0 (Kreidberg+2014)'
    # plotter = Plotter(filenames=[f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_logZpos_Kreidberg_reduction.hdf5", 
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_logZpos_freeTiO_Kreidberg_reduction.hdf5",
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_logZpos_freeVO_Kreidberg_reduction.hdf5",
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_pychegp_logZpos_freeTiOVO_Kreidberg_reduction.hdf5"
    #                             ])

    # planet = 'WASP-74 b'
    # plotter = Plotter(filenames=[f"{file_path}/WASP74b/WASP74b_retrieval_pychegp.hdf5",
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_pychegp_logZpos.hdf5", 
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_pychegp_logZpos_freeTiO.hdf5",
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_pychegp_logZpos_freeVO.hdf5",
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_pychegp_logZpos_freeTiOVO.hdf5",
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_pychegp_logZneg.hdf5", 
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_pychegp_logZneg_freeTiO.hdf5",
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_pychegp_logZneg_freeVO.hdf5",
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_pychegp_logZneg_freeTiOVO.hdf5"
    #                              ])


    # planet = 'WASP-77 A b'
    # plotter = Plotter(filenames=[f"{file_path}/WASP77Ab/WASP77Ab_retrieval_pychegp.hdf5", 
    #                              f"{file_path}/WASP77Ab/WASP77Ab_retrieval_pychegp_freeTiO_photodissociation.hdf5",
    #                              f"{file_path}/WASP77Ab/WASP77Ab_retrieval_pychegp_freeVO_photodissociation.hdf5",
    #                              f"{file_path}/WASP77Ab/WASP77Ab_retrieval_pychegp_freeTiOVO_photodissociation.hdf5"
    #                             ])
    






    # ################################# TRANSMISSION ################################# 
    # mode = 'transmission'
    # params = ['planet_radius','C_O_ratio', 'log_Kzz', 'log_metallicity', 'T_surface', 'T_point1', 'T_point2', 'T_point3','T_top','T']
    # param_title = [r'$R_{planet}$',r'C/O','log(Kzz)', 'log(Z)',r'T$_{surface}$', r'T$_{1}$',r'T$_{2}$', r'T$_{3}$', r'T$_{top}$', r'T$_{isothermal}$']

    # legend_tag = ['FRECKLL, 5-point temperature profile', 'FRECKLL, isothermal temperature profile']
    
    # planet = 'HD-189733 b'
    # plotter = Plotter(filenames=[f"{file_path}/HD189733b/HD189733b_retrieval_transmission_pychegp.hdf5", 
    #                              f"{file_path}/HD189733b/HD189733b_retrieval_transmission_pychegp_isothermal.hdf5"])
    
    # planet = 'HD-209458 b'
    # plotter = Plotter(filenames=[f"{file_path}/HD209458b/HD209458b_retrieval_pychegp_transmission.hdf5", 
    #                              f"{file_path}/HD209458b/HD209458b_retrieval_pychegp_transmission_photodiss_isothermal.hdf5"])
    
    # planet = 'WASP-43 b'
    # plotter = Plotter(filenames=[f"{file_path}/WASP43b/WASP43b_retrieval_transmission_pychegp.hdf5", 
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_transmission_pychegp_isothermal.hdf5"])  

    # planet = 'WASP-74 b'
    # plotter = Plotter(filenames=[f"{file_path}/WASP74b/WASP74b_retrieval_transmission_pychegp.hdf5", 
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_transmission_pychegp_isothermal_maximumTime5only.hdf5"
    #                              ])    






    plotter.plot_temperature_profiles()
    plotter.plot_fitted_spectrum(resolution=10000)
    #plotter.plot_fitted_spectrum()
    plotter.plot_feature_distributions()
    plotter.integrated_plot()


    ################################# POPULATION ################################# 
    # mode = 'population_emission'
    # legend_tag = ['HAT-P-2 b', 'HD-189733 b', 'HD-209458 b', 'Kepler-13 A b', 'TrES-3 b', 'WASP-19 b', 'WASP-4 b', 'WASP-43 b', 'WASP-74 b', 'WASP-77 A b' ]  
    # plotter = Plotter(filenames=[f"{file_path}/HATP2b/HATP2b_retrieval_pychegp_logZneg.hdf5",
    #                              f"{file_path}/HD189733b/HD189733b_retrieval_pychegp_logZneg.hdf5",
    #                              f"{file_path}/HD209458b/HD209458b_retrieval_pychegp.hdf5",
    #                              f"{file_path}/Kepler13Ab/Kepler13Ab_retrieval_pychegp.hdf5",
    #                              f"{file_path}/TrES3b/TrES3b_retrieval_pychegp_photodissociation.hdf5",
    #                              f"{file_path}/WASP19b/WASP19b_retrieval_pychegp_photodissociation_logZneg.hdf5",
    #                              f"{file_path}/WASP4b/WASP4b_retrieval_pychegp.hdf5",
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_pychegp.hdf5",
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_pychegp_logZpos.hdf5",
    #                              f"{file_path}/WASP77Ab/WASP77Ab_retrieval_pychegp_freeTiO_photodissociation.hdf5"
    #                              ])
    
    # mode = 'population_emission_FRECKLLseul'
    # legend_tag = ['HAT-P-2 b', 'HD-189733 b', 'HD-209458 b', 'Kepler-13 A b', 'TrES-3 b', 'WASP-19 b', 'WASP-4 b', 'WASP-43 b', 'WASP-74 b', 'WASP-77 A b' ]
    # plotter = Plotter(filenames=[f"{file_path}/HATP2b/HATP2b_retrieval_pychegp_logZneg.hdf5",
    #                              f"{file_path}/HD189733b/HD189733b_retrieval_pychegp_logZneg.hdf5",
    #                              f"{file_path}/HD209458b/HD209458b_retrieval_pychegp.hdf5",
    #                              f"{file_path}/Kepler13Ab/Kepler13Ab_retrieval_pychegp.hdf5",
    #                              f"{file_path}/TrES3b/TrES3b_retrieval_pychegp_photodissociation.hdf5",
    #                              f"{file_path}/WASP19b/WASP19b_retrieval_pychegp_photodissociation_logZneg.hdf5",
    #                              f"{file_path}/WASP4b/WASP4b_retrieval_pychegp.hdf5",
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_pychegp.hdf5",
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_pychegp_logZpos.hdf5",
    #                              f"{file_path}/WASP77Ab/WASP77Ab_retrieval_pychegp.hdf5"
    #                              ])
    
    # mode = 'population_transmission'
    # legend_tag = ['HD-189733 b', 'HD-209458 b','WASP-43 b', 'WASP-74 b'] 
    # plotter = Plotter(filenames=[f"{file_path}/HD189733b/HD189733b_retrieval_transmission_pychegp_isothermal.hdf5",
    #                              f"{file_path}/HD209458b/HD209458b_retrieval_pychegp_transmission_photodiss_isothermal.hdf5",
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_transmission_pychegp_isothermal.hdf5",
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_transmission_pychegp_isothermal_maximumTime5only.hdf5"
    #                              ])

    # mode = 'population_transmission_5pointTemp'
    # legend_tag = ['HD-189733 b', 'HD-209458 b','WASP-43 b', 'WASP-74 b'] 
    # plotter = Plotter(filenames=[f"{file_path}/HD189733b/HD189733b_retrieval_transmission_pychegp.hdf5",
    #                              f"{file_path}/HD209458b/HD209458b_retrieval_pychegp_transmission.hdf5",
    #                              f"{file_path}/WASP43b/WASP43b_retrieval_transmission_pychegp.hdf5",
    #                              f"{file_path}/WASP74b/WASP74b_retrieval_transmission_pychegp.hdf5"
    #                              ])
    

    # plotter.multi_spectrum_plot(resolution = 10000)
    # plotter.CO_Z_plot()
    # plotter.Teq_CO_plot()
    # plotter.Teq_Z_plot()
    
    # if mode == 'population_emission':
    #     plotter.LogFree_Temp_plot()
    
    # elif mode == 'population_transmission':
    #     plotter.IsoT_Rp_plot()
    #     plotter.CO_ZpZstar_plot()
    
    