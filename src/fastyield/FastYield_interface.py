# import FastYield modules
from .config import instrus, instrus_with_systematics, SNR_thresh, ignore_reflected_thresh_um, get_planet_table_path, filter_type_default
from .get_specs import get_config_data, get_wa
from .spectrum import get_spectrum_contribution_name_model, get_thermal_reflected_spectrum
from .FastCurves import FastCurves
from .FastYield_helpers import get_filename_table, get_planet_index, get_valid_mask, get_SNR_from_table, get_size_from_SNR

# import astropy modules
from astropy import units as u
from astropy.table import QTable
from astropy.coordinates import EarthLocation, AltAz, get_body, SkyCoord
from astropy.time import Time

# import matplotlib modules
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# import numpy modules
import numpy as np

# import other modules
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, DoubleVar, StringVar, Label, Button, Entry, CENTER
from matplotlib.colors import Normalize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ttkwidgets.autocomplete import AutocompleteEntry
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
import traceback


main_color      = "#6F8FAF"  # steel blue
secondary_color = "#111827"  # dark slate
panel_color     = "#E2E6EA"  # cool light grey
entry_color     = "#F5F7FA"  # cool off-white
sedgecolors     = secondary_color
logo_path       = Path(__file__).resolve().parent / "FastYield_logo.png"
logo_height     = 72



# Scatter parameters
s0          = 100
ds          = 3*s0
slinewidths = 0.6
salpha      = 0.95



def load_logo_image(path, height=80, bg_threshold=40):
    logo = Image.open(path).convert("RGBA")
    arr  = np.array(logo)
    mask = np.any(arr[..., :3] > bg_threshold, axis=-1)
    if np.any(mask):
        y, x        = np.where(mask)
        logo        = logo.crop((x.min(), y.min(), x.max() + 1, y.max() + 1))
        arr         = np.array(logo)
        mask        = np.any(arr[..., :3] > bg_threshold, axis=-1)
        arr[..., 3] = np.where(mask, arr[..., 3], 0).astype(np.uint8)
        logo        = Image.fromarray(arr)
    width = int(logo.width * height / logo.height)
    logo  = logo.resize((width, height), Image.LANCZOS)
    return ImageTk.PhotoImage(logo)



def format_quantity(value, decimals=1):
    """Format a scalar (possibly an Astropy Quantity), including missing values."""
    number = getattr(value, "value", value)
    if np.ma.is_masked(number) or not np.isfinite(number):
        return "—"
    unit = getattr(value, "unit", None)
    suffix = f" {unit}" if unit is not None and str(unit) else ""
    return f"{number:.{decimals}f}{suffix}"



class MyWindow(tk.Tk): # https://koor.fr/Python/Tutoriel_Scipy_Stack/matplotlib_integration_ihm.wp

    def __init__(self):
        super().__init__()
        self.table                  = "Archive"
        self.instru                 = "HARMONI"
        self.apodizer               = "NO_SP"
        self.strehl                 = "JQ1"
        self.coronagraph            = None
        self.units                  = "Observational"
        self.exposure_time          = DoubleVar(value=120)
        self.min_elevation          = DoubleVar(value=30)
        self.thermal_model          = "auto"
        self.reflected_model        = "auto"
        self.model_planet           = self.thermal_model+"+"+self.reflected_model
        self.spectrum_contributions = "thermal+reflected"
        self.band                   = "INSTRU"
        self.only_visible_targets   = False
        self.systematics            = False
        self.PCA                    = False
        self.planet_table           = None
        self.calculation            = None
        self.planet                 = None
        self._selected_marker       = None
        self._snr_scatter_artists    = []
        self._detection_text        = None
        self._parameter_title       = None
        self._parameter_table1      = None
        self._parameter_table2      = None
        self._drawn_table_path      = None
        self._drawn_table_mtime_ns  = None
        self._drawn_min_elevation   = None
        self.planet_name            = StringVar()
        self.date                   = datetime.now().strftime("%d/%m/%Y")
        self.date_obs               = StringVar(value=datetime.now().strftime("%d/%m/%Y"))
        self.title("FastYield")
        
        self.Rc          = 100
        self.filter_type = filter_type_default
        self.N_PCA       = 20
        
        try:
            self.state('zoomed') # works fine on Windows!
        except:  
            # Obtenir la taille de l'écran
            screen_width  = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            # Configurer la fenêtre pour qu'elle utilise toute la taille de l'écran
            self.geometry(f"{screen_width}x{screen_height}+0+0")
        self.configure(bg=secondary_color)   

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TCombobox", fieldbackground=entry_color, background=panel_color, foreground=secondary_color, arrowcolor=secondary_color)
        style.map("TCombobox",       fieldbackground=[("readonly", entry_color)], selectbackground=[("readonly", entry_color)], selectforeground=[("readonly", secondary_color)])
        style.configure("TEntry",    fieldbackground=entry_color, foreground=secondary_color)
        
        try:
            self.logo_photo = load_logo_image(logo_path, height=logo_height)
            Label(self, image=self.logo_photo, bg=secondary_color).pack(pady=(4, 6))
        except Exception:
            Label(self, text="FastYield", font='Magneto 30 bold', bg=secondary_color, fg=main_color).pack()

        # CHOIX DE LA TABLE
        self.button_table = tk.Frame(self, bg=secondary_color) ; self.button_table.pack(side=tk.TOP, fill=tk.X)
        tables_dir        = get_planet_table_path(self.table).parent
        tables            = sorted(p.name for p in tables_dir.iterdir() if p.is_dir() and (p / f"{p.name}_Pull_For_FastYield.ecsv").is_file())
        for i, table in enumerate(tables):
            btn = tk.Button(self.button_table, text=table.replace("_", " ").upper(), command=lambda table=table: self.btn_table_clicked(table), bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold'))
            btn.grid(row=0, column=i, sticky="nsew") ; self.button_table.grid_columnconfigure(i, weight=1)

        # CHOIX DE L'INSTRUMENT
        self.button_instru = tk.Frame(self, bg=secondary_color) ; self.button_instru.pack(side=tk.TOP, fill=tk.X)
        for i, instru in enumerate(instrus):
            btn = tk.Button(self.button_instru, text=instru.replace("_", " ").upper(), command=lambda instru=instru: self.btn_instru_clicked(instru), bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold'))
            btn.grid(row=0, column=i, sticky="nsew") ; self.button_instru.grid_columnconfigure(i, weight=1)

        # CHOIX DES UNITES + texp  + name
        self.button_units_texp_name = tk.Frame(self, bg=secondary_color) ; self.button_units_texp_name.pack(side=tk.TOP, fill=tk.X)
        self.__btn_physic           = tk.Button(self.button_units_texp_name, text="Physical units", command=self.btn_physical_clicked, bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_physic.grid(row=0, column=0, sticky="nsew") ; self.button_units_texp_name.grid_columnconfigure(0, weight=1)
        self.__btn_contrast         = tk.Button(self.button_units_texp_name, text="Observational units", command=self.btn_observational_clicked, bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_contrast.grid(row=0, column=3, sticky="nsew") ; self.button_units_texp_name.grid_columnconfigure(3, weight=1)
        self.button_texp            = tk.LabelFrame(self.button_units_texp_name, bg=panel_color) ; self.button_texp.grid(row=0, column=1, sticky="nsew") ; self.button_texp.grid_columnconfigure(1, weight=1)
        self.__btn_exp              = Label(self.button_texp,  text = "Exposure time (in mn):", bg=panel_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_exp.grid(column=0, row=0)
        self.__btn_exp              = Button(self.button_texp, text = "Enter", width=10, height=1, command=self.update_exposure_time_plot, bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_exp.grid(column=2, row=0) 
        self.__btn_exp_entry        = Entry(self.button_texp, width=10, textvariable=self.exposure_time, justify=CENTER, bg=entry_color, fg=secondary_color, insertbackground=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_exp_entry.grid(column=1, row=0)
        self.__btn_exp_entry.bind("<Return>", lambda _: self.update_exposure_time_plot())
        
        # On instancie le Canvas MPL.
        self.__fig = Figure(constrained_layout=True, dpi=75)        
        self.__canvas = FigureCanvasTkAgg(self.__fig, master=self)
        self.__canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.__canvas.mpl_connect("button_press_event", self.canvas_clicked)        
        self.__plt = self.__fig.add_subplot(1, 4, (1, 3))
        self.__plt.tick_params(axis='both', labelsize=14)
        self.cmap = plt.get_cmap("rainbow")
        self.norm = Normalize(0., SNR_thresh)
        sm        = ScalarMappable(norm=self.norm, cmap=self.cmap)
        sm.set_array([])
        self.__cbar = self.__fig.colorbar(sm, ax=self.__plt, pad=0.02, shrink=0.8, orientation='vertical')
        self.__cbar.set_label('S/N', fontsize=20, labelpad=15, rotation=270)
        self.__cbar.ax.tick_params(labelsize=14)
        self.__fig.set_constrained_layout_pads(w_pad=0.4, h_pad=0.4, wspace=0., hspace=0.)
        self.__plt2 = self.__fig.add_subplot(1, 4, 4)
        self.__plt2.set_frame_on(False)
        self.__plt2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        # On initie la table de planète raw (afin d'avoir la liste de tous les noms des planètes)
        self.planet_table = load_planet_table_cached("Archive_Pull_For_FastYield.ecsv")
        
        # Entrée du nom de la planète
        self.button_name      = tk.LabelFrame(self.button_units_texp_name, bg=panel_color) ; self.button_name.grid(row=0, column=2, sticky="nsew") ; self.button_name.grid_columnconfigure(2, weight=1)
        self.__btn_name       = Label(self.button_name, text = "Planet name:", bg=panel_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_name.grid(column=0, row=0)
        self.__btn_name       = Button(self.button_name, text = "Enter", width=10, height=1, command=self.enter_planet_name, bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_name.grid(column=2, row=0) 
        self.__btn_name_entry = AutocompleteEntry(self.button_name, width=20, font=('Arial', 12, 'bold'), completevalues=list(self.planet_table["PlanetName"]), textvariable=self.planet_name, justify=CENTER) ; self.__btn_name_entry.grid(column=1, row=0)
        self.__btn_name_entry.bind("<Return>", lambda _: self.enter_planet_name())
    
        # On initie le plot
        self.btn_instru_clicked()
        
    def enter_planet_name(self):
        if self.planet_name.get() in self.planet_table["PlanetName"]:
            self.planet_index = get_planet_index(planet_table=self.planet_table, planet_name=self.planet_name.get())
            self.planet       = self.planet_table[self.planet_index]
        else:
            self.planet = None
        self.update_selected_planet()
            
    def get_coords(self):
        self.latitude  = self.config_data["latitude"]  # °N
        self.longitude = self.config_data["longitude"] # °E
        self.altitude  = self.config_data["altitude"]  # m

    def btn_table_clicked(self, table):
        self.table = table
        self.planet = None
        self.planet_name.set("")
        self.only_visible_targets = False
        self.__btn_name_entry.set_completion_list(list(load_planet_table_cached(f"{table}_Pull_For_FastYield.ecsv", table=table)["PlanetName"]))
        self.btn_instru_clicked()

    def btn_instru_clicked(self, instru=None):
        if instru is not None:
            self.instru = instru
        self.destroy_lower_buttons()
        self.config_data = get_config_data(self.instru)
        # Do not keep a hidden visibility filter when its controls are absent.
        if self.table != "Archive" or self.config_data["base"] != "ground":
            self.only_visible_targets = False
        self.apodizer    = [apodizer for apodizer in self.config_data["apodizers"]][0]
        self.strehl      = self.config_data["strehls"][0]
        self.coronagraph = self.config_data["coronagraphs"][0]
        self.systematics = False
        self.PCA         = False
        self.thermal_model                             = "auto"
        self.reflected_model                           = "auto" if self.config_data["lambda_range"]["lambda_max"] < ignore_reflected_thresh_um else "None"
        self.spectrum_contributions, self.model_planet = get_spectrum_contribution_name_model(self.thermal_model, self.reflected_model)
        self.band = "INSTRU"
        self.create_button_band()
        self.create_button_calculation()
        if self.table == "Archive":
            self.create_button_model()
            if self.config_data["base"]=="ground":
                self.create_button_visisble_targets()
            elif self.config_data["base"]=="space":
                self.only_visible_targets = False
            if self.instru in instrus_with_systematics:
                self.create_button_systematics()
        self.draw_table_plot()
            

    def create_button_systematics(self):
        self.button_systematics = tk.Frame(self, bg=secondary_color) ; self.button_systematics.pack(side=tk.TOP, fill=tk.X)
        self.__btn = tk.Button(self.button_systematics, text="Without systematics",    command=self.btn_no_systematics_clicked, bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=0, sticky="nsew") ; self.button_systematics.grid_columnconfigure(0, weight=1)
        self.__btn = tk.Button(self.button_systematics, text="With systematics",       command=self.btn_systematics_clicked, bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=1, sticky="nsew") ; self.button_systematics.grid_columnconfigure(1, weight=1)
        self.__btn = tk.Button(self.button_systematics, text="With systematics + PCA", command=self.btn_systematics_PCA_clicked, bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=2, sticky="nsew") ; self.button_systematics.grid_columnconfigure(2, weight=1)

    def btn_no_systematics_clicked(self):
        self.systematics = False
        self.PCA         = False
        self.draw_table_plot()
    def btn_systematics_clicked(self):
        self.systematics = True
        self.PCA         = False
        self.draw_table_plot()
    def btn_systematics_PCA_clicked(self):
        self.systematics = True
        self.PCA         = True
        self.draw_table_plot()
        
    def create_button_band(self):
        self.button_band = tk.Frame(self, bg=secondary_color) ; self.button_band.pack(side=tk.TOP, fill=tk.X)
        list_bands       = ["INSTRU", *self.config_data["gratings"]]
        self.button_bands = tk.LabelFrame(self.button_band, bg=panel_color) ; self.button_bands.grid(row=0, column=0, sticky="nsew") ; self.button_band.grid_columnconfigure(0, weight=1)
        self.__txt_bands  = Label(self.button_bands, text = "Bandwidth: ", bg=panel_color, fg=secondary_color, font=('Arial', 14, 'bold')) ; self.__txt_bands.grid(column=0, row=0) ; self.button_bands.grid_columnconfigure(0, weight=1)
        self.__list_bands = ttk.Combobox(self.button_bands, state='readonly', font=('Arial', 14, 'bold'), justify='center') ; self.__list_bands.grid(column=1, row=0) ; self.button_bands.grid_columnconfigure(1, weight=1)
        popdown = self.__list_bands.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_bands)
        self.__list_bands.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_bands['font'])
        self.__list_bands['values'] = list_bands
        self.__list_bands.current(0) #index de l'élément sélectionné
        self.__list_bands.bind("<<ComboboxSelected>>", lambda _: self.band_clicked())
        list_apodizers = [apodizer for apodizer in self.config_data["apodizers"]]
        list_apodizers = ["None" if elem == "NO_SP" else elem for elem in list_apodizers]
        self.button_apodizers = tk.LabelFrame(self.button_band, bg=panel_color) ; self.button_apodizers.grid(row=0, column=1, sticky="nsew") ; self.button_band.grid_columnconfigure(1, weight=1)
        self.__txt_apodizers  = Label(self.button_apodizers, text = "Apodizer: ", bg=panel_color, fg=secondary_color, font=('Arial', 14, 'bold')) ; self.__txt_apodizers.grid(column=0, row=0) ; self.button_apodizers.grid_columnconfigure(0, weight=1)
        self.__list_apodizers = ttk.Combobox(self.button_apodizers, state='readonly', font=('Arial', 14, 'bold'), justify='center') ; self.__list_apodizers.grid(column=1, row=0) ; self.button_apodizers.grid_columnconfigure(1, weight=1)
        popdown = self.__list_apodizers.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_apodizers)
        self.__list_apodizers.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_apodizers['font'])
        self.__list_apodizers['values'] = list_apodizers
        self.__list_apodizers.current(0) #index de l'élément sélectionné
        self.__list_apodizers.bind("<<ComboboxSelected>>", lambda _: self.apodizer_clicked())
        list_strehls_raw = [strehl for strehl in self.config_data["strehls"]]
        list_strehls     = ["None" if elem == "NO_JQ" else elem for elem in list_strehls_raw]
        self.button_strehls = tk.LabelFrame(self.button_band, bg=panel_color) ; self.button_strehls.grid(row=0, column=2, sticky="nsew") ; self.button_band.grid_columnconfigure(2, weight=1)
        self.__txt_strehls  = Label(self.button_strehls, text = "Strehl: ", bg=panel_color, fg=secondary_color, font=('Arial', 14, 'bold')) ; self.__txt_strehls.grid(column=0, row=0) ; self.button_strehls.grid_columnconfigure(0, weight=1)
        self.__list_strehls = ttk.Combobox(self.button_strehls, state='readonly', font=('Arial', 14, 'bold'), justify='center') ; self.__list_strehls.grid(column=1, row=0) ; self.button_strehls.grid_columnconfigure(1, weight=1)
        popdown = self.__list_strehls.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_strehls)
        self.__list_strehls.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_strehls['font'])
        self.__list_strehls['values'] = list_strehls
        self.__list_strehls.current([i for i, x in enumerate(list_strehls_raw) if x == self.strehl][0]) # index de l'élément sélectionné
        self.__list_strehls.bind("<<ComboboxSelected>>", lambda _: self.strehl_clicked())
        list_coronagraphs_raw = [coronagraph for coronagraph in self.config_data["coronagraphs"]]
        list_coronagraphs     = ["None" if elem == None else elem for elem in list_coronagraphs_raw]
        self.button_coronagraphs = tk.LabelFrame(self.button_band, bg=panel_color) ; self.button_coronagraphs.grid(row=0, column=3, sticky="nsew") ; self.button_band.grid_columnconfigure(3, weight=1)
        self.__txt_coronagraphs  = Label(self.button_coronagraphs, text = "coronagraph: ", bg=panel_color, fg=secondary_color, font=('Arial', 14, 'bold')) ; self.__txt_coronagraphs.grid(column=0, row=0) ; self.button_coronagraphs.grid_columnconfigure(0, weight=1)
        self.__list_coronagraphs = ttk.Combobox(self.button_coronagraphs, state='readonly', font=('Arial', 14, 'bold'), justify='center') ; self.__list_coronagraphs.grid(column=1, row=0) ; self.button_coronagraphs.grid_columnconfigure(1, weight=1)
        popdown = self.__list_coronagraphs.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_coronagraphs)
        self.__list_coronagraphs.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_coronagraphs['font'])
        self.__list_coronagraphs['values'] = list_coronagraphs
        self.__list_coronagraphs.current([i for i, x in enumerate(list_coronagraphs_raw) if x == self.coronagraph][0]) # index de l'élément sélectionné
        self.__list_coronagraphs.bind("<<ComboboxSelected>>", lambda _: self.coronagraph_clicked())
    def band_clicked(self):
        self.band = self.__list_bands.get()
        self.draw_table_plot()
    def apodizer_clicked(self):
        self.apodizer = self.__list_apodizers.get().replace("None", "NO_SP")
        self.draw_table_plot()
    def strehl_clicked(self):
        self.strehl = self.__list_strehls.get().replace("None", "NO_JQ")
        self.draw_table_plot()
    def coronagraph_clicked(self):
        self.coronagraph = self.__list_coronagraphs.get()
        if self.coronagraph == "None":
            self.coronagraph = None
        self.draw_table_plot()
        
    
    def create_button_calculation(self):
        self.button_calculation             = tk.Frame(self, bg=secondary_color) ; self.button_calculation.pack(side=tk.TOP, fill=tk.X)
        self.SNR_calculation_button         = Button(self.button_calculation, text="S/N curves",      command=lambda *args:self.SNR_calculation(), bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.SNR_calculation_button.grid(row=0, column=0, sticky="nsew")  ; self.button_calculation.grid_columnconfigure(0, weight=1)
        self.contrast_calculation_button    = Button(self.button_calculation, text="Contrast curves", command=lambda *args:self.contrast_calculation(), bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.contrast_calculation_button.grid(row=0, column=1, sticky="nsew")  ; self.button_calculation.grid_columnconfigure(1, weight=1)
        self.corner_plot_calculation_button = Button(self.button_calculation, text="Corner plot",     command=lambda *args:self.corner_plot_calculation(), bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.corner_plot_calculation_button.grid(row=0, column=2, sticky="nsew")  ; self.button_calculation.grid_columnconfigure(2, weight=1)


    def create_button_model(self):
        self.button_model         = tk.Frame(self, bg=secondary_color) ; self.button_model.pack(side=tk.TOP, fill=tk.X)
        self.button_thermal_model = tk.LabelFrame(self.button_model, bg=panel_color) ; self.button_thermal_model.grid(row=0, column=0, sticky="nsew") ; self.button_model.grid_columnconfigure(0, weight=1)
        self.__txt_thermal_model  = Label(self.button_thermal_model, text = "Thermal contribution: ", bg=panel_color, fg=secondary_color, font=('Arial', 14, 'bold')) ; self.__txt_thermal_model.grid(column=0, row=0) ; self.button_thermal_model.grid_columnconfigure(0, weight=1)
        self.__list_thermal_model = ttk.Combobox(self.button_thermal_model, state='readonly', font=('Arial', 14, 'bold'), justify='center') ; self.__list_thermal_model.grid(column=1, row=0) ; self.button_thermal_model.grid_columnconfigure(1, weight=1)
        popdown = self.__list_thermal_model.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_thermal_model)
        self.__list_thermal_model.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_thermal_model['font'])
        self.__list_thermal_model['values'] = ("Off", "On")
        self.__list_thermal_model.current(1) #index de l'élément sélectionné
        self.__list_thermal_model.bind("<<ComboboxSelected>>", lambda _: self.contribution_clicked())
        self.button_reflected_model = tk.LabelFrame(self.button_model, bg=panel_color) ; self.button_reflected_model.grid(row=0, column=1, sticky="nsew") ; self.button_model.grid_columnconfigure(1, weight=1)
        self.__txt_reflected_model  = Label(self.button_reflected_model, text = "Reflected contribution: ", bg=panel_color, fg=secondary_color, font=('Arial', 14, 'bold')) ; self.__txt_reflected_model.grid(column=0, row=0) ; self.button_reflected_model.grid_columnconfigure(0, weight=1)
        self.__list_reflected_model = ttk.Combobox(self.button_reflected_model, state='readonly', font=('Arial', 14, 'bold'), justify='center') ; self.__list_reflected_model.grid(column=1, row=0) ; self.button_reflected_model.grid_columnconfigure(1, weight=1)
        popdown = self.__list_reflected_model.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_reflected_model)
        self.__list_reflected_model.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_reflected_model['font'])
        if self.config_data["lambda_range"]["lambda_max"] < ignore_reflected_thresh_um:
            self.__list_reflected_model['values'] = ("Off", "On")
            self.__list_reflected_model.current(1) #index de l'élément sélectionné
        else:
            self.__list_reflected_model['values'] = ("Off")
            self.__list_reflected_model.current(0) #index de l'élément sélectionné
        self.__list_reflected_model.bind("<<ComboboxSelected>>", lambda _: self.contribution_clicked())
    def contribution_clicked(self):
        self.thermal_model   = "auto" if (self.__list_thermal_model.get()   == "On") else "None"
        self.reflected_model = "auto" if (self.__list_reflected_model.get() == "On") else "None"
        self.spectrum_contributions, self.model_planet = get_spectrum_contribution_name_model(self.thermal_model, self.reflected_model)
        self.draw_table_plot()
        
    def create_button_visisble_targets(self):
        # CHOIX DES UNITES + texp
        self.button_visible_targets = tk.Frame(self, bg=secondary_color) ; self.button_visible_targets.pack(side=tk.TOP, fill=tk.X)
        self.__btn_all_targets      = tk.Button(self.button_visible_targets, text="All targets", command=self.btn_all_targets_clicked, bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_all_targets.grid(row=0, column=0, sticky="nsew") ; self.button_visible_targets.grid_columnconfigure(0, weight=1)
        self.__btn_visible_targets  = tk.Button(self.button_visible_targets, text="Only visible targets from the observation site", command=self.btn_visible_targets_clicked, bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_visible_targets.grid(row=0, column=1, sticky="nsew") ; self.button_visible_targets.grid_columnconfigure(1, weight=1)
        self.button_elevation_min      = tk.LabelFrame(self.button_visible_targets, bg=panel_color) ; self.button_elevation_min.grid(row=0, column=2, sticky="nsew") ; self.button_elevation_min.grid_columnconfigure(2, weight=1)
        self.__txt_elevation_min       = Label(self.button_elevation_min, text = "Min elevation (in °): ", bg=panel_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__txt_elevation_min.grid(column=0, row=0)
        self.__btn_elevation_min       = Button(self.button_elevation_min, text = "Enter", width=10, height=1, command=self.btn_visible_targets_clicked, bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_elevation_min.grid(column=2, row=0) 
        self.__btn_entry_elevation_min = Entry(self.button_elevation_min, width=10, textvariable=self.min_elevation, justify=CENTER, bg=entry_color, fg=secondary_color, insertbackground=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_entry_elevation_min.grid(column=1, row=0) 
        self.__btn_entry_elevation_min.bind("<Return>", lambda _: self.btn_visible_targets_clicked())
        self.button_visibility_curve      = tk.LabelFrame(self.button_visible_targets, bg=panel_color) ; self.button_visibility_curve.grid(row=0, column=3, sticky="nsew") ; self.button_visibility_curve.grid_columnconfigure(3, weight=1)
        self.__txt_visibility_curve       = Label(self.button_visibility_curve, text = "Visibility on (dd/mm/yyyy): ", bg=panel_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__txt_visibility_curve.grid(column=0, row=0)
        self.__btn_visibility             = Button(self.button_visibility_curve, text = "Enter", width=10, height=1, command=self.btn_visibility, bg=main_color, fg=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_visibility.grid(column=3, row=0) 
        self.__btn_entry_visibility_curve = Entry(self.button_visibility_curve, width=10, textvariable=self.date_obs, justify=CENTER, bg=entry_color, fg=secondary_color, insertbackground=secondary_color, font=('Arial', 12, 'bold')) ; self.__btn_entry_visibility_curve.grid(column=1, row=0) 
        self.__btn_entry_visibility_curve.bind("<Return>", lambda _: self.btn_visibility())

    def btn_all_targets_clicked(self):
        self.only_visible_targets = False
        self.draw_table_plot()
    def btn_visible_targets_clicked(self):
        self.only_visible_targets = True
        self.draw_table_plot()
    def btn_visibility(self):
        if self.planet is not None:
            location        = EarthLocation(lat=self.latitude*u.deg, lon=self.longitude*u.deg, height=self.altitude*u.m)
            tf              = TimezoneFinder()
            timezone_str    = tf.timezone_at(lng=self.longitude, lat=self.latitude)
            local_tz        = pytz.timezone(timezone_str)
            target          = SkyCoord(ra=self.planet["RA"], dec=self.planet["Dec"])       
            date_obs_local  = local_tz.localize(datetime.strptime(self.date_obs.get(), "%d/%m/%Y"))    
            midnight        = date_obs_local.replace(hour=0, minute=0, second=0)
            delta_midnight  = np.linspace(-12, 12, 500) * u.hour  # 24-hour period divided into intervals
            times_utc       = Time(midnight) + delta_midnight
            times_local     = [t.to_datetime(pytz.utc).astimezone(local_tz) for t in times_utc]
            frame           = AltAz(obstime=times_utc, location=location)
            target_altaz    = target.transform_to(frame)
            sun_altaz       = get_body("sun", time=times_utc, location=location).transform_to(frame)
            moon_altaz      = get_body("moon", time=times_utc, location=location).transform_to(frame)
            is_night        = sun_altaz.alt < 0   * u.deg
            twilight        = sun_altaz.alt < -6  * u.deg  # Crépuscule astronomique
            deep_twilight   = sun_altaz.alt < -12 * u.deg  # Crépuscule profond
            deep_night      = sun_altaz.alt < -18 * u.deg  # Nuit astronomique
            very_deep_night = sun_altaz.alt < -24 * u.deg  # Nuit très profonde            
            plt.figure(figsize=(12, 7), dpi=300)
            ax = plt.gca()            
            ax.axhline(0, color='black', linewidth=2)
            ax.axhline(self.min_elevation.get(), color='black', linestyle=":", linewidth=2, label="Min elevation")            
            ax.fill_between(times_local, -90, 90, where=is_night,        color='#264653', alpha=0.10, zorder=0)
            ax.fill_between(times_local, -90, 90, where=twilight,        color='#264653', alpha=0.25, zorder=0)
            ax.fill_between(times_local, -90, 90, where=deep_twilight,   color='#264653', alpha=0.45, zorder=0)
            ax.fill_between(times_local, -90, 90, where=deep_night,      color='#264653', alpha=0.65, zorder=0)
            ax.fill_between(times_local, -90, 90, where=very_deep_night, color='#264653', alpha=0.85, zorder=0)            
            ax.plot(times_local, target_altaz.alt, label=f'{self.planet["PlanetName"]}', color='#2A9D8F', linewidth=2.4, zorder=5)
            ax.plot(times_local, moon_altaz.alt,   label='Moon', color='#6C757D', linestyle='--', linewidth=1.6, zorder=4)
            ax.plot(times_local, sun_altaz.alt,    label='Sun',  color='#E76F51', linestyle='--', linewidth=1.6, zorder=3)            
            ax.set_title(f"Visibility of {self.planet['PlanetName']} with {self.instru} on {self.date_obs.get()}\n (RA: {self.planet['RA'].value:.2f}°, DEC: {self.planet['Dec'].value:.2f}°)", fontsize=16, pad=15, weight='bold')
            ax.set_xlabel('Local Time', fontsize=14)
            ax.set_ylabel('Elevation (°)', fontsize=14)
            ax.set_xlim(times_local[0], times_local[-1])
            ax.set_ylim(-90, 90)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=local_tz))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.tick_params(axis='x', labelrotation=0, labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.4)
            ax.minorticks_on()            
            ax.legend(loc='lower center', fancybox=True, shadow=False, ncol=4, fontsize=14, frameon=True, edgecolor="gray", facecolor="whitesmoke")
            plt.tight_layout()
            plt.show()


        
    def btn_physical_clicked(self):
        self.units = "Physical"
        self.draw_table_plot()
    def btn_observational_clicked(self):
        self.units = "Observational"
        self.draw_table_plot()
        
    def valid_exposure_time(self):
        """Reject invalid exposure times before evaluating or plotting S/N."""
        try:
            exposure_time = self.exposure_time.get()
        except tk.TclError:
            exposure_time = np.nan
        if not np.isfinite(exposure_time) or exposure_time <= 0:
            messagebox.showwarning("FastYield", "Exposure time must be a positive number (in minutes).")
            return False
        return True

    def draw_table_plot(self):
        if not self.valid_exposure_time():
            return
        # Full redraw: old Matplotlib artists are no longer valid.
        self.__plt.clear()
        self.__plt2.clear()
        self.__plt2.set_axis_off()
        self._parameter_title    = None
        self._parameter_table1   = None
        self._parameter_table2   = None
        self._selected_marker    = None
        self._snr_scatter_artists = []
        self._detection_text     = None
        self._drawn_table_path   = None
        
        # Loading the SNR catalog from the selected table directory.
        filename          = get_filename_table(table=self.table, instru=self.instru, apodizer=self.apodizer, strehl=self.strehl, coronagraph=self.coronagraph, systematics=self.systematics, PCA=self.PCA, name_model=self.model_planet)
        path = get_planet_table_path(self.table) / filename
        if not path.is_file():
            self.planet_table = load_planet_table_cached(f"{self.table}_Pull_For_FastYield.ecsv", table=self.table)
            self.planet = None
            self.planet_index = None
            self.planet_name.set("")
            self.SNR = np.array([])
            self._xi = self._yi = np.array([])
            self._click_indices = np.array([], dtype=int)
            self.__plt.text(0.5, 0.5, f"S/N table not available for {self.table} / {self.instru}:\n{filename}", ha="center", va="center", transform=self.__plt.transAxes, fontsize=14)
            self.__canvas.draw_idle()
            return
        self.planet_table = load_planet_table_cached(filename, table=self.table)
        self._drawn_table_path     = path
        self._drawn_table_mtime_ns = path.stat().st_mtime_ns
        self._drawn_min_elevation  = self.min_elevation.get()
        
        # Filtrage des planètes non-visibles depuis le site d'observation (if necessary)
        if self.config_data["base"]=="ground":
            self.get_coords()
        if self.only_visible_targets: # on filtre les planètes non-visibles depuis le site d'observation
            self.planet_table = self.planet_table[(90 - np.abs(self.latitude-self.planet_table["Dec"].value))>self.min_elevation.get()]
            # # Observable planets according to the input date
            # obs_mask          = are_planets_observable(latitude=self.latitude, longitude=self.longitude, altitude=self.altitude, planet_table=self.planet_table, date_obs=self.date_obs.get(), min_elevation=self.min_elevation.get())
            # self.planet_table = self.planet_table[obs_mask]
        
        # Filtrage des planètes sans Masse
        if self.units == "Physical":
            self.planet_table = self.planet_table[get_valid_mask(self.planet_table["PlanetMass"])]
        
        # WORKING ANGLE
        iwa, owa = get_wa(instru=self.instru, band=self.band, sep_unit="mas")
        
        # calcul du SNR de chaque planète sur la bande
        self.SNR                                 = get_SNR_from_table(planet_table=self.planet_table, exposure_time=self.exposure_time.get(), band=self.band)
        self.z_instru                            = np.copy(self.SNR)
        self.z_instru[self.z_instru>=SNR_thresh] = SNR_thresh
        nb_detected                              = len(self.SNR[self.SNR>=SNR_thresh])
        
        # Definition des vecteurs x, y et z
        if self.units == "Observational":
            self.x_instru = self.planet_table["AngSep"]
            if self.band != "INSTRU":
                self.y_instru  = 10**(-(self.planet_table["Planet"+self.band+"mag("+self.spectrum_contributions+")"]-np.array(self.planet_table["Star"+self.band+"mag"]))/2.5)*u.dimensionless_unscaled
            else:
                self.y_instru  = 10**(-(self.planet_table["Planet"+self.band+"mag("+self.instru+")("+self.spectrum_contributions+")"]-np.array(self.planet_table["Star"+self.band+"mag("+self.instru+")"]))/2.5)*u.dimensionless_unscaled
        elif self.units == "Physical":
            self.x_instru = self.planet_table['SMA']
            self.y_instru = self.planet_table["PlanetMass"]
        if "DiscoveryMethod" in self.planet_table.colnames:
            label_rv_nep = "Radial Velocity"
            label_im_jup = "Direct Imaging"
            label_tr_ter = "Transit"
            label_ot_sol = "Astrometry" if np.all(self.planet_table["DiscoveryMethod"] == "Astrometry") else "Other"
            mask_instru_rv_nep = self.planet_table["DiscoveryMethod"] == "Radial Velocity"
            mask_instru_im_jup = self.planet_table["DiscoveryMethod"] == "Imaging"
            mask_instru_tr_ter = self.planet_table["DiscoveryMethod"] == "Transit"
            mask_instru_ot_sol = (self.planet_table["DiscoveryMethod"]!="Imaging") & (self.planet_table["DiscoveryMethod"]!="Radial Velocity") & (self.planet_table["DiscoveryMethod"]!="Transit")
        else:
            label_rv_nep = "Neptunian worlds"
            label_im_jup = "Jovian worlds"
            label_tr_ter = "Terran worlds"
            label_ot_sol = "Stellar worlds"
            mask_instru_rv_nep = self.planet_table["PlanetType"] == "Neptunian"
            mask_instru_im_jup = self.planet_table["PlanetType"] == "Jovian"
            mask_instru_tr_ter = self.planet_table["PlanetType"] == "Terran"
            mask_instru_ot_sol = self.planet_table["PlanetType"] == "Stellar"
        x_instru_rv_nep = self.x_instru[mask_instru_rv_nep].value
        x_instru_im_jup = self.x_instru[mask_instru_im_jup].value
        x_instru_tr_ter = self.x_instru[mask_instru_tr_ter].value
        x_instru_ot_sol = self.x_instru[mask_instru_ot_sol].value
        y_instru_rv_nep = self.y_instru[mask_instru_rv_nep].value
        y_instru_im_jup = self.y_instru[mask_instru_im_jup].value
        y_instru_tr_ter = self.y_instru[mask_instru_tr_ter].value
        y_instru_ot_sol = self.y_instru[mask_instru_ot_sol].value
        z_instru_rv_nep = self.z_instru[mask_instru_rv_nep]
        z_instru_im_jup = self.z_instru[mask_instru_im_jup]
        z_instru_tr_ter = self.z_instru[mask_instru_tr_ter]
        z_instru_ot_sol = self.z_instru[mask_instru_ot_sol]
        
        # For canvas_clicked
        self._build_click_index()
                
        # PLOT 
        self.__plt.plot([], [], c='black', ls="", marker='v', ms=10, label=label_tr_ter)
        self.__plt.plot([], [], c='black', ls="", marker='o', ms=10, label=label_rv_nep)
        self.__plt.plot([], [], c='black', ls="", marker='s', ms=10, label=label_im_jup)
        self.__plt.plot([], [], c='black', ls="", marker='P', ms=10, label=label_ot_sol)
        scatter_kwargs = dict(cmap=self.cmap, norm=self.norm, edgecolors=sedgecolors, linewidths=slinewidths, alpha=salpha, zorder=3)
        self._snr_scatter_artists = [
            (self.__plt.scatter(x_instru_tr_ter, y_instru_tr_ter, s=get_size_from_SNR(SNR=self.SNR[mask_instru_tr_ter], s0=s0, ds=ds, SNR_min=0, SNR_max=1_000*SNR_thresh), c=z_instru_tr_ter, marker="v", **scatter_kwargs), mask_instru_tr_ter),
            (self.__plt.scatter(x_instru_rv_nep, y_instru_rv_nep, s=get_size_from_SNR(SNR=self.SNR[mask_instru_rv_nep], s0=s0, ds=ds, SNR_min=0, SNR_max=1_000*SNR_thresh), c=z_instru_rv_nep, marker="o", **scatter_kwargs), mask_instru_rv_nep),
            (self.__plt.scatter(x_instru_im_jup, y_instru_im_jup, s=get_size_from_SNR(SNR=self.SNR[mask_instru_im_jup], s0=s0, ds=ds, SNR_min=0, SNR_max=1_000*SNR_thresh), c=z_instru_im_jup, marker="s", **scatter_kwargs), mask_instru_im_jup),
            (self.__plt.scatter(x_instru_ot_sol, y_instru_ot_sol, s=get_size_from_SNR(SNR=self.SNR[mask_instru_ot_sol], s0=s0, ds=ds, SNR_min=0, SNR_max=1_000*SNR_thresh), c=z_instru_ot_sol, marker="P", **scatter_kwargs), mask_instru_ot_sol),
        ]
        if self.band == "INSTRU":
            self.lmin = self.config_data["lambda_range"]["lambda_min"] ; self.lmax = self.config_data["lambda_range"]["lambda_max"]
            self.R = 0.
            for band in self.config_data["gratings"]:
                self.R += self.config_data["gratings"][band].R/len(self.config_data["gratings"]) # mean resolution
        else:
            self.lmin = self.config_data['gratings'][self.band].lmin ; self.lmax = self.config_data['gratings'][self.band].lmax
            self.R = self.config_data["gratings"][self.band].R
        self.update_plot_title()
        if self.units == "Observational":
            self.__plt.set_xlabel(f'Maximum angular separation (in {self.x_instru.unit})', fontsize=18)
            self.__plt.set_ylabel(f'Planet-to-star contrast (on {self.band}-band)', fontsize=18)
            self.__plt.axvspan(iwa, owa, color='black', alpha=0.3, lw=0, label="Working angle", zorder=2)
            if "thermal" in self.spectrum_contributions:
                self.__plt.set_ylim(1e-11, 1)
            else:
                self.__plt.set_ylim(1e-14, 1e-3)
            self.__plt.set_xlim(1e-2, 1e6)
        elif self.units == "Physical":
            self.__plt.set_xlabel(f'Semi Major Axis (in {self.x_instru.unit})', fontsize=18)
            self.__plt.set_ylabel(f"Planet mass (in {self.y_instru.unit})", fontsize=18)
            self.__plt.set_ylim(1e-1, 2.5e4) ; self.__plt.set_xlim(5e-3, 1e4)
        self.__plt.legend(loc="lower right", fontsize=16)
        self.__plt.text(.01, .99, f'Total number of planets: {len(self.planet_table)}', ha='left', va='top', transform=self.__plt.transAxes, fontsize=12)
        self._detection_text = self.__plt.text(.01, .965, f'Number of planets detected: {nb_detected} / {len(self.planet_table)}', ha='left', va='top', transform=self.__plt.transAxes, fontsize=12)
        self._selected_marker = self.__plt.plot([], [], "kX", ms=14, zorder=4)[0]
        self.update_selected_planet(redraw=False)
        self.__plt.set_yscale('log')
        self.__plt.set_xscale('log')
        self.__plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.__plt.minorticks_on()
        self.__canvas.draw_idle()
    
    def update_plot_title(self):
        """Update only the title, keeping the current instrument and band."""
        txt_syst = "(with systematics+PCA)" if self.PCA else "(with systematics)"
        if not self.systematics:
            txt_syst = "(without systematics)"
        catalog_label = "Known exoplanets" if self.table == "Archive" else self.table.replace("_", " ") + " planets"
        spectral_range = f"from {self.lmin:.1f} to {self.lmax:.1f} µm"
        if self.config_data["type"] != "imager":
            spectral_range += f" with R ≈ {int(round(self.R, -2))}"
        self.__plt.set_title(
            f'{catalog_label} detection yield with {self.instru} ({self.spectrum_contributions} light) '
            f'\n on {self.band}-band ({spectral_range}) with '
            + r'$t_{exp}$=' + str(round(self.exposure_time.get())) + 'mn ' + txt_syst,
            fontsize=18,
        )

    def update_exposure_time_plot(self):
        """Update S/N colors/sizes without rebuilding the catalogue or scatter plot."""
        if not self.valid_exposure_time():
            return

        # Rebuild if the ECSV has changed, or a pending visibility input was edited.
        path = self._drawn_table_path
        if (path is None or not path.is_file()
                or path.stat().st_mtime_ns != self._drawn_table_mtime_ns
                or (self.only_visible_targets and self.min_elevation.get() != self._drawn_min_elevation)
                or len(self._snr_scatter_artists) != 4):
            self.draw_table_plot()
            return

        self.SNR      = get_SNR_from_table(planet_table=self.planet_table, exposure_time=self.exposure_time.get(), band=self.band)
        self.z_instru = np.copy(self.SNR)
        self.z_instru[self.z_instru >= SNR_thresh] = SNR_thresh
        for artist, mask in self._snr_scatter_artists:
            artist.set_array(self.z_instru[mask])
            artist.set_sizes(get_size_from_SNR(SNR=self.SNR[mask], s0=s0, ds=ds, SNR_min=0, SNR_max=1_000*SNR_thresh))

        nb_detected = np.count_nonzero(self.SNR >= SNR_thresh)
        self._detection_text.set_text(f'Number of planets detected: {nb_detected} / {len(self.planet_table)}')
        self.update_plot_title()
        if self.planet is not None:
            if self._parameter_table2 is not None:
                self._parameter_table2[(0, 0)].get_text().set_text(format_quantity(self.SNR[self.planet_index], 1))
            else:
                self.draw_table_parameters()
        self.__canvas.draw_idle()

    def update_selected_planet(self, redraw=True):
        """Update the selection marker/panel; the thousands of points are unchanged."""
        if self._selected_marker is None:
            if redraw:
                self.draw_table_plot()
            return

        # Keep the same auto-reload behavior when an ECSV is replaced on disk.
        path = self._drawn_table_path
        if redraw and path is not None and (not path.is_file() or path.stat().st_mtime_ns != self._drawn_table_mtime_ns):
            self.draw_table_plot()
            return

        self._selected_marker.set_data([], [])
        if self.planet is not None:
            matching = np.flatnonzero(self.planet_table["PlanetName"] == self.planet["PlanetName"])
            if len(matching):
                self.planet_index = int(matching[0])
                self.planet       = self.planet_table[self.planet_index]
                self.draw_table_parameters()
                self._selected_marker.set_data([self.x_instru[self.planet_index].value], [self.y_instru[self.planet_index].value])
            else:
                self.planet = None
                self.planet_index = None
                self.planet_name.set("")
        if self.planet is None:
            self.__plt2.clear()
            self.__plt2.set_axis_off()
            self._parameter_title  = None
            self._parameter_table1 = None
            self._parameter_table2 = None
        if redraw:
            self.__canvas.draw_idle()

    def _build_click_index(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            self._xi = np.log10(np.asarray(self.x_instru.value, dtype=float))
            self._yi = np.log10(np.asarray(self.y_instru.value, dtype=float))
        self._click_indices = np.flatnonzero(np.isfinite(self._xi) & np.isfinite(self._yi))

    def canvas_clicked(self, event):
        if event.inaxes is not self.__plt or event.xdata is None or event.ydata is None:
            return
        if not np.isfinite(event.xdata) or not np.isfinite(event.ydata) or event.xdata <= 0 or event.ydata <= 0 or not len(self._click_indices):
            return
        lx, ly            = np.log10(event.xdata), np.log10(event.ydata)
        idx               = np.square(self._xi[self._click_indices] - lx) + np.square(self._yi[self._click_indices] - ly)
        planet_index      = int(self._click_indices[np.argmin(idx)])
        if self.planet is not None and self.planet_index == planet_index:
            path = self._drawn_table_path
            if path is not None and path.is_file() and path.stat().st_mtime_ns == self._drawn_table_mtime_ns:
                return
        self.planet_index = planet_index
        self.planet       = self.planet_table[self.planet_index]
        self.planet_name.set(self.planet["PlanetName"])
        self.update_selected_planet()
            
    def draw_table_parameters(self):
        title = f"{self.planet['PlanetName']} ({self.planet['PlanetType']})\n on {self.band}-band of {self.instru}"
        if self._parameter_title is None:
            self._parameter_title = self.__plt2.text(0.5, 1, title, fontsize=22, weight='bold', bbox=dict(facecolor='white', edgecolor=secondary_color, boxstyle='round,pad=0.5', alpha=0.9), va='center', ha='center', zorder=10)
        else:
            self._parameter_title.set_text(title)
        if self.band == "INSTRU":
            self.mag_planet = self.planet["Planet"+self.band+"mag("+self.instru+")("+self.spectrum_contributions+")"].value
            self.mag_star   = self.planet["Star"+self.band+"mag("+self.instru+")"].value
        else:
            self.mag_planet = self.planet["Planet"+self.band+"mag("+self.spectrum_contributions+")"].value
            self.mag_star   = self.planet["Star"+self.band+"mag"].value
        self.flux_ratio = 10**(-(self.mag_planet-self.mag_star)/2.5)
        data1 =    [[ 'Planet', 'Star'],
                    [ 'T',    format_quantity(self.planet["PlanetTeff"], 0),           format_quantity(self.planet["StarTeff"], 0)],
                    [ 'lg',   format_quantity(self.planet["PlanetLogg"], 1),           format_quantity(self.planet["StarLogg"], 1)],
                    [ 'M',    format_quantity(self.planet["PlanetMass"], 1),           format_quantity(self.planet["StarMass"], 1)],
                    [ 'R',    format_quantity(self.planet["PlanetRadius"], 1),         format_quantity(self.planet["StarRadius"], 1)],
                    [ 'mag',  format_quantity(self.mag_planet, 1),                      format_quantity(self.mag_star, 1)],
                    [ 'RV',   format_quantity(self.planet["PlanetRadialVelocity"], 1), format_quantity(self.planet["StarRadialVelocity"], 1)],
                    [ 'Vrot', format_quantity(self.planet["PlanetVrot"], 1),           format_quantity(self.planet["StarVrot"], 1)]]
        column_headers1 = data1.pop(0)
        row_headers1    = [row.pop(0) for row in data1]
        cell_text1      = [row for row in data1]
        row_colors1     = [main_color] * len(row_headers1)
        col_colors1     = [main_color] * len(column_headers1)
        if self._parameter_table1 is None:
            table1 = self.__plt2.table(cellText=cell_text1, rowLabels=row_headers1, rowColours=row_colors1, rowLoc='center', colColours=col_colors1, colLabels=column_headers1, loc='center', bbox=[0.12, 0., 0.89, 0.4])
            table1.set_fontsize(16)
            for cell in table1.get_celld().values():
                cell.set_text_props(ha="center", va="center")
                cell.set_edgecolor(secondary_color)
                cell.set_linewidth(0.5)
            self._parameter_table1 = table1
        else:
            for i, row in enumerate(cell_text1, start=1):
                for j, value in enumerate(row):
                    self._parameter_table1[(i, j)].get_text().set_text(value)
        data2 =  [[ 'S/N',                 format_quantity(self.SNR[self.planet_index], 1)],
                  [ 'Flux ratio',          f'{self.flux_ratio:.1e}' ],
                  [ 'Max. sep.',           format_quantity(self.planet['AngSep'], 0)],
                  [ 'SMA',                 format_quantity(self.planet['SMA'], 1)],
                  [ 'Discovery method',    self.planet['DiscoveryMethod']],
                  [ 'Star spectral type',  f"{self.planet['StarSpT']}"],
                  [ 'Star age',            format_quantity(self.planet['StarAge'], 2)],
                  [ 'Inclination',         format_quantity(self.planet['Inc'], 0)],
                  [ 'Distance',            format_quantity(self.planet['Distance'], 1)]]
        row_headers2 = [row.pop(0) for row in data2]
        cell_text2   = [row for row in data2]
        row_colors2  = [main_color] * len(row_headers2)
        if self._parameter_table2 is None:
            table2 = self.__plt2.table(cellText=cell_text2, rowLabels=row_headers2, rowColours=row_colors2, rowLoc='center', loc='center', bbox=[0.49, 0.47, 0.52, 0.45])
            table2.set_fontsize(16)
            for cell in table2.get_celld().values():
                cell.set_text_props(ha="center", va="center")
                cell.set_edgecolor(secondary_color)
                cell.set_linewidth(0.5)
            self._parameter_table2 = table2
            self.__plt2.text(1.05, -0.05, self.date, horizontalalignment='right', size=14, weight='light')
        else:
            for i, row in enumerate(cell_text2):
                self._parameter_table2[(i, 0)].get_text().set_text(row[0])
        


    def SNR_calculation(self):
        self.calculation = "SNR"
        self.FastCurves_calculation()
    
    def contrast_calculation(self):
        self.calculation = "contrast"
        self.FastCurves_calculation()
    
    def corner_plot_calculation(self):
        self.calculation = "corner plot"
        self.FastCurves_calculation()
    
    def FastCurves_calculation(self):
        if self.planet is None:
            messagebox.showinfo("FastYield", "Select a planet before running FastCurves.")
            return
        if not self.valid_exposure_time():
            return
        if self.calculation == "corner plot" and self.config_data["type"] == "imager":
            messagebox.showinfo("FastYield", "Corner plots are not implemented for imagers.")
            return
        # Parameters retrieval (corner plot)
        if self.calculation == "corner plot":
            
            # Picking the band with the highest S/N
            if self.band == "INSTRU":
                snr_max          = 0
                band_only        = None
                planet_table_one = self.planet_table[self.planet_index:self.planet_index+1]
                for band in self.config_data["gratings"]:
                    snr = get_SNR_from_table(planet_table=planet_table_one, exposure_time=self.exposure_time.get(), band=band)[0]
                    if np.isfinite(snr) and snr > snr_max:
                        snr_max   = snr
                        band_only = band
                if band_only is None:
                    print(f"No valid S/N available for {self.planet['PlanetName']}.")
                    return
            else:
                band_only = self.band
            
        # Load spectra only after confirming the requested calculation is possible.
        planet_spectrum, planet_thermal, planet_reflected, star_spectrum = get_thermal_reflected_spectrum(planet=self.planet, thermal_model=self.thermal_model, reflected_model=self.reflected_model, instru=self.instru, wave_model=None, wave_K=None, counts_vega_K=None, show=True)
        mag_p = self.planet[f"PlanetINSTRUmag({self.instru})({self.spectrum_contributions})"].value
        mag_s = self.planet[f"StarINSTRUmag({self.instru})"].value
        band0 = "instru"

        if self.calculation == "corner plot":
            # Choosing the model for the retrieval
            if self.spectrum_contributions == "thermal+reflected":
                if self.planet[f"Planet{band_only}mag(thermal)"].value < self.planet[f"Planet{band_only}mag(reflected)"].value:
                    planet_spectrum.model = planet_thermal.model
                else:
                    planet_spectrum.model = planet_reflected.model
            elif self.spectrum_contributions == "thermal":
                planet_spectrum.model = planet_thermal.model
            else:
                planet_spectrum.model = planet_reflected.model
                
            # FastCurves calculation
            try:
                FastCurves(instru=self.instru, band_only=band_only, calculation=self.calculation, mag_star=mag_s, band0=band0, exposure_time=self.exposure_time.get(), mag_planet=mag_p, separation_planet=self.planet["AngSep"].value/1000, return_FastYield=False, show_plot=True, verbose=True, planet_name=self.planet["PlanetName"], planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, apodizer=self.apodizer, strehl=self.strehl, coronagraph=self.coronagraph, Rc=self.Rc, filter_type=self.filter_type, systematics=self.systematics, PCA=self.PCA, N_PCA=self.N_PCA)
            except Exception as e:
                print(f"Corner plot failed for {self.planet['PlanetName']}: {e}")
                traceback.print_exc()
        
        # S/N or contrast plots
        else:
            if self.band == "INSTRU":
                band_only = None 
            else:
                band_only = self.band
            FastCurves(instru=self.instru, band_only=band_only, calculation=self.calculation, mag_star=mag_s, band0=band0, exposure_time=self.exposure_time.get(), mag_planet=mag_p, separation_planet=self.planet["AngSep"].value/1000, return_FastYield=False, show_plot=True, verbose=True, planet_name=self.planet["PlanetName"], planet_spectrum=planet_spectrum, star_spectrum=star_spectrum, apodizer=self.apodizer, strehl=self.strehl, coronagraph=self.coronagraph, Rc=self.Rc, filter_type=self.filter_type, systematics=self.systematics, PCA=self.PCA, N_PCA=self.N_PCA)

    def destroy_lower_buttons(self):
        for name in ("button_band", "button_calculation", "button_model", "button_systematics", "button_visible_targets"):
            widget = getattr(self, name, None)
            if widget is not None and widget.winfo_exists():
                widget.destroy()



@lru_cache(maxsize=4)
def _load_planet_table_cached(path, mtime_ns):
    return QTable.read(path, format="ascii.ecsv")



def load_planet_table_cached(filename: str, table="Archive"):
    """Cache catalogs by path and modification time (reload updated ECSV files)."""
    path = get_planet_table_path(table) / filename
    return _load_planet_table_cached(path, path.stat().st_mtime_ns)

    

def FastYield_interface():
    app = MyWindow()
    app.mainloop()