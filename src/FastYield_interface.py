import sys
from math import *
import tkinter as tk
from tkinter import *
from tkinter import ttk
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ttkwidgets.autocomplete import AutocompleteEntry
from timezonefinder import TimezoneFinder
import pytz
from datetime import datetime, timedelta  # Make sure to import timedelta
import matplotlib.dates as mdates
from joblib import Parallel, delayed

sys.path.insert(0, '../src')
from src.FastYield import *

class MyWindow(tk.Tk): # https://koor.fr/Python/Tutoriel_Scipy_Stack/matplotlib_integration_ihm.wp

    def __init__(self):
        super().__init__()
        self.table                  = "Archive"
        self.instru                 = "HARMONI"
        self.apodizer               = "NO_SP"
        self.strehl                 = "JQ1"
        self.units                  = "Observational"
        self.exposure_time          = DoubleVar(value=120)
        self.min_elevation          = DoubleVar(value=30)
        self.thermal_model          = "BT-Settl"
        self.reflected_model        = "PICASO"
        self.model_planet           = self.thermal_model+"+"+self.reflected_model
        self.spectrum_contributions = "thermal+reflected"
        self.band                   = "INSTRU"
        self.only_visible_targets   = False
        self.systematics            = False
        self.PCA                    = False
        self.planet_table           = None
        self.popup                  = None
        self.popup_state            = "Close"
        self.calculation            = None
        self.planet                 = None
        self.planet_name            = StringVar()
        self.date                   = datetime.now().strftime("%d/%m/%Y")
        self.date_obs               = StringVar(value=datetime.now().strftime("%d/%m/%Y"))
        self.title("FastYield")
        
        try:
            self.state('zoomed') # works fine on Windows!
        except:  
            # Obtenir la taille de l'écran
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            # Configurer la fenêtre pour qu'elle utilise toute la taille de l'écran
            self.geometry(f"{screen_width}x{screen_height}+0+0")
        self.configure(bg='black')   
        
        Label(self, text="FastYield", font='Magneto 30 bold', bg="black", fg="dark orange").pack()
        
        # CHOIX DE LA TABLE
        self.button_table = tk.Frame(self) ; self.button_table.pack(side=tk.TOP, fill=tk.X)
        self.__btn_archive = tk.Button(self.button_table, text="ARCHIVE TABLE", command=self.btn_archive_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_archive.grid(row=0, column=0, sticky="nsew") ; self.button_table.grid_columnconfigure(0, weight=1)
        self.__btn_simulated = tk.Button(self.button_table, text="SIMULATED TABLE", command=self.btn_simulated_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_simulated.grid(row=0, column=1, sticky="nsew") ; self.button_table.grid_columnconfigure(1, weight=1)

        # CHOIX DE L'INSTRUMENT 
        self.button_instru   = tk.Frame(self) ; self.button_instru.pack(side=tk.TOP, fill=tk.X)
        self.__btn_harmoni   = tk.Button(self.button_instru, text="ELT/HARMONI", command=self.btn_harmoni_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_harmoni.grid(row=0, column=0, sticky="nsew") ; self.button_instru.grid_columnconfigure(0, weight=1)
        self.__btn_andes     = tk.Button(self.button_instru, text="ELT/ANDES", command=self.btn_andes_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_andes.grid(row=0, column=1, sticky="nsew") ; self.button_instru.grid_columnconfigure(1, weight=1)
        self.__btn_eris      = tk.Button(self.button_instru, text="VLT/ERIS", command=self.btn_eris_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_eris.grid(row=0, column=2, sticky="nsew") ; self.button_instru.grid_columnconfigure(2, weight=1)
        self.__btn_hirise    = tk.Button(self.button_instru, text="VLT/HiRISE", command=self.btn_hirise_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_hirise.grid(row=0, column=3, sticky="nsew") ; self.button_instru.grid_columnconfigure(3, weight=1)
        self.__btn_mirimrs   = tk.Button(self.button_instru, text="JWST/MIRI/MRS", command=self.btn_mirimrs_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_mirimrs.grid(row=0, column=4, sticky="nsew") ; self.button_instru.grid_columnconfigure(4, weight=1)
        self.__btn_nirspec   = tk.Button(self.button_instru, text="JWST/NIRSpec/IFU", command=self.btn_nirspec_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_nirspec.grid(row=0, column=5, sticky="nsew") ; self.button_instru.grid_columnconfigure(5, weight=1)
        self.__btn_nircam    = tk.Button(self.button_instru, text="JWST/NIRCam", command=self.btn_nircam_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_nircam.grid(row=0, column=6, sticky="nsew") ; self.button_instru.grid_columnconfigure(6, weight=1)
        self.__btn_vipapyrus = tk.Button(self.button_instru, text="OHP/VIPAPYRUS", command=self.btn_vipapyrus_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_vipapyrus.grid(row=0, column=7, sticky="nsew") ; self.button_instru.grid_columnconfigure(7, weight=1)

        # CHOIX DES UNITES + texp  + name
        self.button_units_texp_name = tk.Frame(self) ; self.button_units_texp_name.pack(side=tk.TOP, fill=tk.X)
        self.__btn_physic = tk.Button(self.button_units_texp_name, text="Physical units", command=self.btn_physical_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_physic.grid(row=0, column=0, sticky="nsew") ; self.button_units_texp_name.grid_columnconfigure(0, weight=1)
        self.__btn_contrast = tk.Button(self.button_units_texp_name, text="Observational units", command=self.btn_observational_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_contrast.grid(row=0, column=3, sticky="nsew") ; self.button_units_texp_name.grid_columnconfigure(3, weight=1)
        self.button_texp = tk.LabelFrame(self.button_units_texp_name) ; self.button_texp.grid(row=0, column=1, sticky="nsew") ; self.button_texp.grid_columnconfigure(1, weight=1)
        self.__btn_exp = Label(self.button_texp, text = "Exposure time (in mn):", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_exp.grid(column=0, row=0)
        self.__btn_exp = Button(self.button_texp, text = "Enter", width=10, height=1, command=self.draw_table_plot, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_exp.grid(column=2, row=0) 
        self.__btn_exp_entry = Entry(self.button_texp, width=10, textvariable=self.exposure_time, justify=CENTER, font=('Arial', 12, 'bold')) ; self.__btn_exp_entry.grid(column=1, row=0)
        self.__btn_exp_entry.bind("<Return>", lambda _: self.draw_table_plot())
        
        # On instancie le Canvas MPL.
        self.__fig = plt.figure(constrained_layout=True, dpi=75)        
        self.__canvas = FigureCanvasTkAgg(self.__fig, master=self)
        self.__canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.__canvas.mpl_connect("button_press_event", self.canvas_clicked)        
        self.__plt = self.__fig.add_subplot(1, 4, (1, 3))
        self.__plt.tick_params(axis='both', labelsize=14)
        self.cmap = plt.get_cmap("rainbow")
        self.norm = Normalize(0., 5.)
        sm = ScalarMappable(norm=self.norm, cmap=self.cmap)
        sm.set_array([])
        self.__cbar = self.__fig.colorbar(sm, ax=self.__plt, pad=0.02, shrink=0.8, orientation='vertical')
        self.__cbar.set_label('S/N', fontsize=20, labelpad=15, rotation=270)
        self.__cbar.ax.tick_params(labelsize=14)
        self.__fig.set_constrained_layout_pads(w_pad=0.4, h_pad=0.4, wspace=0., hspace=0.)
        self.__plt2 = self.__fig.add_subplot(1, 4, 4)
        self.__plt2.set_frame_on(False)
        self.__plt2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        # On initie la table de planète raw (afin d'avoir la liste de tous les noms des planètes)
        self.planet_table = load_planet_table("Archive_Pull.ecsv")
        
        # Entrée du nom de la planète
        self.button_name = tk.LabelFrame(self.button_units_texp_name) ; self.button_name.grid(row=0, column=2, sticky="nsew") ; self.button_name.grid_columnconfigure(2, weight=1)
        self.__btn_name = Label(self.button_name, text = "Planet name:", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_name.grid(column=0, row=0)
        self.__btn_name = Button(self.button_name, text = "Enter", width=10, height=1, command=self.enter_planet_name, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_name.grid(column=2, row=0) 
        self.__btn_name_entry = AutocompleteEntry(self.button_name, width=20, font=('Arial', 12, 'bold'), completevalues=list(self.planet_table["PlanetName"]), textvariable=self.planet_name, justify=CENTER) ; self.__btn_name_entry.grid(column=1, row=0)
        self.__btn_name_entry.bind("<Return>", lambda _: self.enter_planet_name())
    
        # On initie le plot
        self.btn_instru_clicked()
        
    def enter_planet_name(self):
        if self.planet_name.get() in self.planet_table["PlanetName"]:
            self.planet_index = get_planet_index(planet_table=self.planet_table, planet_name=self.planet_name.get())
            self.planet = self.planet_table[self.planet_index]
        else:
            self.planet = None
        self.draw_table_plot()
            
    def get_coords(self):
        self.latitude = self.config_data["latitude"] # °N
        self.longitude = self.config_data["longitude"] # °E
        self.altitude = self.config_data["altitude"] # m

    def btn_archive_clicked(self):
        self.destroy_lower_buttons()
        self.table = "Archive"
        self.btn_instru_clicked()
    def btn_simulated_clicked(self):
        self.destroy_lower_buttons()
        self.table = "Simulated"
        self.btn_instru_clicked()
        
    def btn_instru_clicked(self):
        try:
            self.config_data = get_config_data(self.instru)
        except:
            raise KeyError("PLEASE ADD THE INSTRUMENT IN THE FUNCTION !")
        self.destroy_lower_buttons()
        self.apodizer = "NO_SP"
        if self.config_data["base"]=="ground":
            if self.instru == "HARMONI":
                self.strehl = "JQ1"
            elif self.instru == "ANDES" or self.instru=="VIPAPYRUS" or self.instru=="HiRISE":
                self.strehl = "MED"
            elif self.instru == "ERIS":
                self.strehl = "JQ0"
        elif self.config_data["base"]=="space":
            self.strehl = "NO_JQ"
        self.systematics = False
        self.PCA = False
        if self.config_data["lambda_range"]["lambda_max"] < 6:
            self.thermal_model = "BT-Settl"
            self.reflected_model = "PICASO"
            self.model_planet = self.thermal_model+"+"+self.reflected_model
            self.spectrum_contributions = "thermal+reflected"
        else:
            self.model_planet = "BT-Settl"
            self.thermal_model = "BT-Settl"
            self.reflected_model = "None"
            self.spectrum_contributions = "thermal"
        self.band = "INSTRU"
        self.create_button_band()
        self.create_button_calculation()
        if self.table == "Archive":
            self.create_button_model()
            if self.config_data["base"]=="ground":
                self.create_button_visisble_targets()
            elif self.config_data["base"]=="space":
                self.only_visible_targets = False
            if self.instru in instru_with_systematics:
                self.create_button_systematics()
        self.draw_table_plot()
            
    def btn_harmoni_clicked(self):
        self.instru = "HARMONI"
        self.btn_instru_clicked()
    def btn_andes_clicked(self):
        self.instru = "ANDES"
        self.btn_instru_clicked()
    def btn_eris_clicked(self):
        self.instru = "ERIS"
        self.btn_instru_clicked()
    def btn_hirise_clicked(self):
        self.instru = "HiRISE"
        self.btn_instru_clicked()
    def btn_mirimrs_clicked(self):
        self.instru = "MIRIMRS"
        self.btn_instru_clicked()
    def btn_nirspec_clicked(self):
        self.instru = "NIRSpec"
        self.btn_instru_clicked()
    def btn_nircam_clicked(self):
        self.instru = "NIRCam"
        self.btn_instru_clicked()
    def btn_vipapyrus_clicked(self):
        self.instru = "VIPAPYRUS"
        self.btn_instru_clicked()
    
    def create_button_systematics(self):
        self.button_systematics = tk.Frame(self) ; self.button_systematics.pack(side=tk.TOP, fill=tk.X)
        self.__btn = tk.Button(self.button_systematics, text="Without systematics", command=self.btn_no_systematics_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=0, sticky="nsew") ; self.button_systematics.grid_columnconfigure(0, weight=1)
        self.__btn = tk.Button(self.button_systematics, text="With systematics", command=self.btn_systematics_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=1, sticky="nsew") ; self.button_systematics.grid_columnconfigure(1, weight=1)
        self.__btn = tk.Button(self.button_systematics, text="With systematics + PCA", command=self.btn_systematics_PCA_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=2, sticky="nsew") ; self.button_systematics.grid_columnconfigure(2, weight=1)

    def btn_no_systematics_clicked(self):
        self.systematics = False
        self.draw_table_plot()
    def btn_systematics_clicked(self):
        self.systematics = True
        self.PCA = False
        self.draw_table_plot()
    def btn_systematics_PCA_clicked(self):
        self.systematics = True
        self.PCA = True
        self.draw_table_plot()
        
    def create_button_band(self):
        self.button_band = tk.Frame(self) ; self.button_band.pack(side=tk.TOP, fill=tk.X)
        list_bands = ["INSTRU"]
        for name_band in self.config_data['gratings']:
            list_bands.append(name_band)
        self.button_bands = tk.LabelFrame(self.button_band) ; self.button_bands.grid(row=0, column=0, sticky="nsew") ; self.button_band.grid_columnconfigure(0, weight=1)
        self.__txt_bands = Label(self.button_bands, text = "Bandwidth: ", fg="black", font=('Arial', 14, 'bold')) ; self.__txt_bands.grid(column=0, row=0) ; self.button_bands.grid_columnconfigure(0, weight=1)
        self.__list_bands = ttk.Combobox(self.button_bands, state='readonly', font=('Arial', 14, 'bold'), justify='center') ; self.__list_bands.grid(column=1, row=0) ; self.button_bands.grid_columnconfigure(1, weight=1)
        popdown = self.__list_bands.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_bands)
        self.__list_bands.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_bands['font'])
        self.__list_bands['values'] = list_bands
        self.__list_bands.current(0) #index de l'élément sélectionné
        self.__list_bands.bind("<<ComboboxSelected>>", lambda _: self.band_clicked())
        list_apodizers = [apodizer for apodizer in self.config_data["apodizers"]]
        list_apodizers = ["None" if elem == "NO_SP" else elem for elem in list_apodizers]
        self.button_apodizers = tk.LabelFrame(self.button_band) ; self.button_apodizers.grid(row=0, column=1, sticky="nsew") ; self.button_band.grid_columnconfigure(1, weight=1)
        self.__txt_apodizers = Label(self.button_apodizers, text = "Apodizer: ", fg="black", font=('Arial', 14, 'bold')) ; self.__txt_apodizers.grid(column=0, row=0) ; self.button_apodizers.grid_columnconfigure(0, weight=1)
        self.__list_apodizers = ttk.Combobox(self.button_apodizers, state='readonly', font=('Arial', 14, 'bold'), justify='center') ; self.__list_apodizers.grid(column=1, row=0) ; self.button_apodizers.grid_columnconfigure(1, weight=1)
        popdown = self.__list_apodizers.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_apodizers)
        self.__list_apodizers.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_apodizers['font'])
        self.__list_apodizers['values'] = list_apodizers
        self.__list_apodizers.current(0) #index de l'élément sélectionné
        self.__list_apodizers.bind("<<ComboboxSelected>>", lambda _: self.apodizer_clicked())
        list_strehls_raw = [strehl for strehl in self.config_data["strehls"]]
        list_strehls     = ["None" if elem == "NO_JQ" else elem for elem in list_strehls_raw]
        self.button_strehls = tk.LabelFrame(self.button_band) ; self.button_strehls.grid(row=0, column=2, sticky="nsew") ; self.button_band.grid_columnconfigure(2, weight=1)
        self.__txt_strehls = Label(self.button_strehls, text = "Strehl: ", fg="black", font=('Arial', 14, 'bold')) ; self.__txt_strehls.grid(column=0, row=0) ; self.button_strehls.grid_columnconfigure(0, weight=1)
        self.__list_strehls = ttk.Combobox(self.button_strehls, state='readonly', font=('Arial', 14, 'bold'), justify='center') ; self.__list_strehls.grid(column=1, row=0) ; self.button_strehls.grid_columnconfigure(1, weight=1)
        popdown = self.__list_strehls.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_strehls)
        self.__list_strehls.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_strehls['font'])
        self.__list_strehls['values'] = list_strehls
        self.__list_strehls.current([i for i, x in enumerate(list_strehls_raw) if x == self.strehl][0]) # index de l'élément sélectionné
        self.__list_strehls.bind("<<ComboboxSelected>>", lambda _: self.strehl_clicked())
    def band_clicked(self):
        self.band = self.__list_bands.get()
        self.draw_table_plot()
    def apodizer_clicked(self):
        self.apodizer = self.__list_apodizers.get().replace("None", "NO_SP")
        self.draw_table_plot()
    def strehl_clicked(self):
        self.strehl = self.__list_strehls.get().replace("None", "NO_JQ")
        self.draw_table_plot()
        
    
    def create_button_calculation(self):
        self.button_calculation = tk.Frame(self) ; self.button_calculation.pack(side=tk.TOP, fill=tk.X)
        self.SNR_calculation_button = Button(self.button_calculation, text="S/N curves", command=lambda *args:self.SNR_calculation(), bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.SNR_calculation_button.grid(row=0, column=0, sticky="nsew")  ; self.button_calculation.grid_columnconfigure(0, weight=1)
        self.contrast_calculation_button = Button(self.button_calculation, text="Contrast curves", command=lambda *args:self.contrast_calculation(), bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.contrast_calculation_button.grid(row=0, column=1, sticky="nsew")  ; self.button_calculation.grid_columnconfigure(1, weight=1)
        self.corner_plot_calculation_button = Button(self.button_calculation, text="Corner plot", command=lambda *args:self.corner_plot_calculation(), bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.corner_plot_calculation_button.grid(row=0, column=2, sticky="nsew")  ; self.button_calculation.grid_columnconfigure(2, weight=1)


    def create_button_model(self):
        self.button_model = tk.Frame(self) ; self.button_model.pack(side=tk.TOP, fill=tk.X)
        self.button_thermal_model = tk.LabelFrame(self.button_model) ; self.button_thermal_model.grid(row=0, column=0, sticky="nsew") ; self.button_model.grid_columnconfigure(0, weight=1)
        self.__txt_thermal_model = Label(self.button_thermal_model, text = "Thermal contribution (atmospheric model): ", fg="black", font=('Arial', 14, 'bold')) ; self.__txt_thermal_model.grid(column=0, row=0) ; self.button_thermal_model.grid_columnconfigure(0, weight=1)
        self.__list_thermal_model = ttk.Combobox(self.button_thermal_model, state='readonly', font=('Arial', 14, 'bold'), justify='center') ; self.__list_thermal_model.grid(column=1, row=0) ; self.button_thermal_model.grid_columnconfigure(1, weight=1)
        popdown = self.__list_thermal_model.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_thermal_model)
        self.__list_thermal_model.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_thermal_model['font'])
        if self.config_data["lambda_range"]["lambda_max"] < 6:
            self.__list_thermal_model['values'] = ("None", "BT-Settl", "Exo-REM", "PICASO")
        else:
            self.__list_thermal_model['values'] = ("None", "BT-Settl", "Exo-REM")
        self.__list_thermal_model.current(1) #index de l'élément sélectionné
        self.__list_thermal_model.bind("<<ComboboxSelected>>", lambda _: self.contribution_clicked())
        self.button_reflected_model = tk.LabelFrame(self.button_model) ; self.button_reflected_model.grid(row=0, column=1, sticky="nsew") ; self.button_model.grid_columnconfigure(1, weight=1)
        self.__txt_reflected_model = Label(self.button_reflected_model, text = "Reflected contribution (albedo model): ", fg="black", font=('Arial', 14, 'bold')) ; self.__txt_reflected_model.grid(column=0, row=0) ; self.button_reflected_model.grid_columnconfigure(0, weight=1)
        self.__list_reflected_model = ttk.Combobox(self.button_reflected_model, state='readonly', font=('Arial', 14, 'bold'), justify='center') ; self.__list_reflected_model.grid(column=1, row=0) ; self.button_reflected_model.grid_columnconfigure(1, weight=1)
        popdown = self.__list_reflected_model.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_reflected_model)
        self.__list_reflected_model.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_reflected_model['font'])
        self.__list_reflected_model['values'] = ("None", "PICASO", "tellurics", "flat")
        if self.config_data["lambda_range"]["lambda_max"] < 6:
            self.__list_reflected_model['values'] = ("None", "PICASO", "tellurics", "flat")
            self.__list_reflected_model.current(1) #index de l'élément sélectionné
        else:
            self.__list_reflected_model['values'] = ("None")
            self.__list_reflected_model.current(0) #index de l'élément sélectionné
        self.__list_reflected_model.bind("<<ComboboxSelected>>", lambda _: self.contribution_clicked())
    def contribution_clicked(self):
        try:
            self.thermal_model = self.__list_thermal_model.get()
        except:
            pass
        try:
            self.reflected_model = self.__list_reflected_model.get()
        except:
            pass
        if self.thermal_model != "None" and self.reflected_model != "None":
            self.spectrum_contributions = "thermal+reflected"
            self.model_planet = self.thermal_model+"+"+self.reflected_model
        elif self.thermal_model != "None" and self.reflected_model == "None":
            self.spectrum_contributions = "thermal"
            if self.thermal_model == "PICASO":
                self.model_planet = self.thermal_model+"_thermal_only"
            else:
                self.model_planet = self.thermal_model
        elif self.thermal_model == "None" and self.reflected_model != "None":
            self.spectrum_contributions = "reflected"
            if self.reflected_model == "PICASO":
                self.model_planet = self.reflected_model+"_reflected_only"
            else:
                self.model_planet = self.reflected_model
        self.draw_table_plot()
        
    def create_button_visisble_targets(self):
        # CHOIX DES UNITES + texp
        self.button_visible_targets = tk.Frame(self) ; self.button_visible_targets.pack(side=tk.TOP, fill=tk.X)
        self.__btn = tk.Button(self.button_visible_targets, text="All targets", command=self.btn_all_targets_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=0, sticky="nsew") ; self.button_visible_targets.grid_columnconfigure(0, weight=1)
        self.__btn = tk.Button(self.button_visible_targets, text="Only visible targets from the observation site", command=self.btn_visible_targets_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=1, sticky="nsew") ; self.button_visible_targets.grid_columnconfigure(1, weight=1)
        self.button_elevation_min = tk.LabelFrame(self.button_visible_targets) ; self.button_elevation_min.grid(row=0, column=2, sticky="nsew") ; self.button_elevation_min.grid_columnconfigure(2, weight=1)
        self.__txt_elevation_min = Label(self.button_elevation_min, text = "Min elevation (in °): ", fg="black", font=('Arial', 12, 'bold')) ; self.__txt_elevation_min.grid(column=0, row=0)
        self.__btn_elevation_min = Button(self.button_elevation_min, text = "Enter", width=10, height=1, command=self.btn_visible_targets_clicked, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_elevation_min.grid(column=2, row=0) 
        self.__btn_entry_elevation_min = Entry(self.button_elevation_min, width=10, textvariable=self.min_elevation, justify=CENTER, font=('Arial', 12, 'bold')) ; self.__btn_entry_elevation_min.grid(column=1, row=0) 
        self.__btn_entry_elevation_min.bind("<Return>", lambda _: self.btn_visible_targets_clicked())
        self.button_visibility_curve = tk.LabelFrame(self.button_visible_targets) ; self.button_visibility_curve.grid(row=0, column=3, sticky="nsew") ; self.button_visibility_curve.grid_columnconfigure(3, weight=1)
        self.__txt_visibility_curve = Label(self.button_visibility_curve, text = "Visibility on (dd/mm/yyyy): ", fg="black", font=('Arial', 12, 'bold')) ; self.__txt_visibility_curve.grid(column=0, row=0)
        self.__btn_visibility = Button(self.button_visibility_curve, text = "Enter", width=10, height=1, command=self.btn_visibility, bg="dark orange", fg="black", font=('Arial', 12, 'bold')) ; self.__btn_visibility.grid(column=3, row=0) 
        self.__btn_entry_visibility_curve = Entry(self.button_visibility_curve, width=10, textvariable=self.date_obs, justify=CENTER, font=('Arial', 12, 'bold')) ; self.__btn_entry_visibility_curve.grid(column=1, row=0) 
        self.__btn_entry_visibility_curve.bind("<Return>", lambda _: self.btn_visibility())

    def btn_all_targets_clicked(self):
        self.only_visible_targets = False
        self.draw_table_plot()
    def btn_visible_targets_clicked(self):
        self.only_visible_targets = True
        self.draw_table_plot()
    def btn_visibility(self):
        if self.only_visible_targets:
            self.draw_table_plot()
        if self.planet is not None:
            location = EarthLocation(lat=self.latitude*u.deg, lon=self.longitude*u.deg, height=self.altitude*u.m)
            tf = TimezoneFinder() ; timezone_str = tf.timezone_at(lng=self.longitude, lat=self.latitude) ; local_tz = pytz.timezone(timezone_str)
            time_utc = Time.now() ; time_local = time_utc.to_datetime(pytz.utc).astimezone(local_tz)        
            target = SkyCoord(ra=float(self.planet["RA"].value)*u.deg, dec=float(self.planet["Dec"].value)*u.deg)       
            date_obs_local = datetime.strptime(self.date_obs.get(), "%d/%m/%Y").astimezone(local_tz)    
            midnight       = date_obs_local.replace(hour=0, minute=0, second=0)
            delta_midnight = np.linspace(-12, 12, 500) * u.hour  # 24-hour period divided into intervals
            times_utc   = Time(midnight) + delta_midnight
            times_local = [t.to_datetime(pytz.utc).astimezone(local_tz) for t in times_utc]
            frame = AltAz(obstime=times_utc, location=location)
            target_altaz = target.transform_to(frame)
            sun_altaz    = get_body("sun", time=times_utc).transform_to(frame)
            moon_altaz   = get_body("moon", time=times_utc).transform_to(frame)
            plt.figure(dpi=300) ; plt.axhline(0, color='black', lw=2) ; plt.axhline(self.min_elevation.get(), color='black', linestyle=":", lw=1, label="min elevation") ; plt.title(f'Visibility of {self.planet["PlanetName"]} with {self.instru} on {self.date_obs.get()} \n (RA: {round(float(self.planet["RA"].value), 2)}°, DEC: {round(float(self.planet["Dec"].value), 2)}°)') ; plt.xlabel('Local Time') ; plt.ylabel('Elevation (°)')
            is_night        = sun_altaz.alt < 0 * u.deg
            twilight        = sun_altaz.alt < -6 * u.deg  # Crépuscule astronomique
            deep_twilight   = sun_altaz.alt < -12 * u.deg  # Crépuscule profond
            deep_night      = sun_altaz.alt < -18 * u.deg  # Nuit astronomique
            very_deep_night = sun_altaz.alt < -24 * u.deg  # Nuit très profonde
            plt.fill_between(times_local, -90, 90, where=is_night, color='#5060ff', alpha=0.1)  # Nuit légère
            plt.fill_between(times_local, -90, 90, where=twilight, color='#5060ff', alpha=0.3)  # Crépuscule
            plt.fill_between(times_local, -90, 90, where=deep_twilight, color='#5060ff', alpha=0.5)  # Crépuscule profond
            plt.fill_between(times_local, -90, 90, where=deep_night, color='#5060ff', alpha=0.7)  # Nuit astronomique
            plt.fill_between(times_local, -90, 90, where=very_deep_night, color='#5060ff', alpha=0.9)  # Nuit très profonde        
            plt.plot(times_local, target_altaz.alt, label=f'{self.planet["PlanetName"]}', color='red', zorder=5)
            plt.plot(times_local, moon_altaz.alt, label='Moon', color='gray', linestyle='--', zorder=4)
            plt.plot(times_local, sun_altaz.alt, label='Sun', color='orange', linestyle='--', zorder=3)        
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=local_tz)) # local time axis
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))  # Ticks tevery 3 hours
            plt.legend() ; plt.xlim(times_local[0], times_local[-1]) ; plt.ylim(-90, 90) ; plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=3) ; plt.minorticks_on() ; plt.show()
        
    def btn_physical_clicked(self):
        self.units = "Physical"
        self.draw_table_plot()
    def btn_observational_clicked(self):
        self.units = "Observational"
        self.draw_table_plot()
        
    def draw_table_plot(self):
        if self.popup is not None:
            self.popup.destroy()
        self.__plt.clear()
        # chargement des tables de planètes
        planet_table_raw = load_planet_table(self.table+"_Pull_for_FastCurves.ecsv")
        if self.table == "Archive":
            planet_table_raw = planet_table_raw[np.logical_not(planet_table_raw['AngSep'].mask)]
        if self.systematics:
            if self.PCA:
                self.planet_table = load_planet_table(self.table+"_Pull_"+self.instru+"_"+self.apodizer+"_"+self.strehl+"_with_systematics+PCA_"+self.model_planet+".ecsv")
            else:
                self.planet_table = load_planet_table(self.table+"_Pull_"+self.instru+"_"+self.apodizer+"_"+self.strehl+"_with_systematics_"+self.model_planet+".ecsv")
        else:
            self.planet_table = load_planet_table(self.table+"_Pull_"+self.instru+"_"+self.apodizer+"_"+self.strehl+"_without_systematics_"+self.model_planet+".ecsv")
        # Filtrage les planètes non-visibles depuis le site d'observation
        if self.config_data["base"]=="ground":
            self.get_coords()
        if self.only_visible_targets: # on filtre les planètes non-visibles depuis le site d'observation
            planet_table_raw = planet_table_raw[(90 - np.abs(self.latitude-planet_table_raw["Dec"].value))>self.min_elevation.get()]
            obs_mask = are_planets_observable(latitude=self.latitude, longitude=self.longitude, altitude=self.altitude, planet_table=self.planet_table, date_obs=self.date_obs.get(), min_elevation=self.min_elevation.get())
            self.planet_table = self.planet_table[obs_mask]
        if self.units == "Physical":
            planet_table_raw  = planet_table_raw[~get_mask(planet_table_raw, "PlanetMass")]
            self.planet_table = self.planet_table[~get_mask(self.planet_table, "PlanetMass")]
            planet_table_raw  = planet_table_raw[~get_mask(planet_table_raw, "SMA")]
            self.planet_table = self.planet_table[~get_mask(self.planet_table, "SMA")]
        # WORKING ANGLE
        iwa, owa          = get_wa(config_data=self.config_data, band=self.band, apodizer=self.apodizer, sep_unit="mas")
        planet_table_raw  = planet_table_raw[planet_table_raw["AngSep"] <= iwa * u.mas]
        self.planet_table = self.planet_table[self.planet_table["AngSep"] >= iwa * u.mas]
        # calcul du SNR de chaque planète sur la bande
        self.SNR = np.copy(np.sqrt(self.exposure_time.get()/self.planet_table['DIT_'+self.band]) * self.planet_table['signal_'+self.band] / np.sqrt( self.planet_table['sigma_fund_'+self.band]**2 + (self.exposure_time.get()/self.planet_table['DIT_'+self.band])*self.planet_table['sigma_syst_'+self.band]**2 ))
        z_instru = np.copy(self.SNR) ; z_instru[z_instru>=5] = 5 ; nb_detected = len(self.SNR[self.SNR>=5])
        # Definition des vecteurs x, y et z
        if self.units == "Observational":
            x_raw = planet_table_raw["AngSep"] ; self.x_instru = self.planet_table["AngSep"]
            if self.band != "INSTRU":
                y_raw = 10**(-(planet_table_raw["Planet"+self.band+"mag("+self.spectrum_contributions+")"]-np.array(planet_table_raw["Star"+self.band+"mag"]))/2.5)*u.dimensionless_unscaled
                self.y_instru  = 10**(-(self.planet_table["Planet"+self.band+"mag("+self.spectrum_contributions+")"]-np.array(self.planet_table["Star"+self.band+"mag"]))/2.5)*u.dimensionless_unscaled
            else:
                y_raw = 10**(-(planet_table_raw["Planet"+self.band+"mag("+self.instru+")("+self.spectrum_contributions+")"]-np.array(planet_table_raw["Star"+self.band+"mag("+self.instru+")"]))/2.5)*u.dimensionless_unscaled
                self.y_instru  = 10**(-(self.planet_table["Planet"+self.band+"mag("+self.instru+")("+self.spectrum_contributions+")"]-np.array(self.planet_table["Star"+self.band+"mag("+self.instru+")"]))/2.5)*u.dimensionless_unscaled
        elif self.units == "Physical":
            x_raw = planet_table_raw['SMA'] ; self.x_instru = self.planet_table['SMA']
            y_raw = planet_table_raw["PlanetMass"] ; self.y_instru = self.planet_table["PlanetMass"]
        if self.table == "Archive":
            mask_raw_rv_nep = planet_table_raw["DiscoveryMethod"] == "Radial Velocity" ; mask_instru_rv_nep =  self.planet_table["DiscoveryMethod"] == "Radial Velocity" ; label_rv_nep = "Radial Velocity"
            mask_raw_im_jup = planet_table_raw["DiscoveryMethod"] == "Imaging" ; mask_instru_im_jup =  self.planet_table["DiscoveryMethod"] == "Imaging" ; label_im_jup = "Direct Imaging"
            mask_raw_tr_ter = planet_table_raw["DiscoveryMethod"] == "Transit" ; mask_instru_tr_ter =  self.planet_table["DiscoveryMethod"] == "Transit" ; label_tr_ter = "Transit"
            mask_raw_ot_sol = (planet_table_raw["DiscoveryMethod"]!="Imaging") & (planet_table_raw["DiscoveryMethod"]!="Radial Velocity") & (planet_table_raw["DiscoveryMethod"]!="Transit") ; mask_instru_ot_sol =  (self.planet_table["DiscoveryMethod"]!="Imaging") & (self.planet_table["DiscoveryMethod"]!="Radial Velocity") & (self.planet_table["DiscoveryMethod"]!="Transit") ; label_ot_sol = "Other"
        elif self.table == "Simulated":
            mask_raw_rv_nep = planet_table_raw["PlanetType"] == "Neptunian" ; mask_instru_rv_nep =  self.planet_table["PlanetType"] == "Neptunian" ; label_rv_nep = "Neptunian worlds"
            mask_raw_im_jup = planet_table_raw["PlanetType"] == "Jovian" ; mask_instru_im_jup =  self.planet_table["PlanetType"] == "Jovian" ; label_im_jup = "Jovian worlds"
            mask_raw_tr_ter = planet_table_raw["PlanetType"] == "Terran" ; mask_instru_tr_ter =  self.planet_table["PlanetType"] == "Terran" ; label_tr_ter = "Terran worlds"
            mask_raw_ot_sol = planet_table_raw["PlanetType"] == "Stellar" ; mask_instru_ot_sol =  self.planet_table["PlanetType"] == "Stellar" ; label_ot_sol = "Stellar worlds"
        x_raw_rv_nep = x_raw[mask_raw_rv_nep].value ; y_raw_rv_nep = y_raw[mask_raw_rv_nep].value
        x_raw_im_jup = x_raw[mask_raw_im_jup].value ; y_raw_im_jup = y_raw[mask_raw_im_jup].value
        x_raw_tr_ter = x_raw[mask_raw_tr_ter].value ; y_raw_tr_ter = y_raw[mask_raw_tr_ter].value
        x_raw_ot_sol = x_raw[mask_raw_ot_sol].value ; y_raw_ot_sol = y_raw[mask_raw_ot_sol].value
        x_instru_rv_nep = self.x_instru[mask_instru_rv_nep].value ; y_instru_rv_nep = self.y_instru[mask_instru_rv_nep].value ; z_instru_rv_nep = z_instru[mask_instru_rv_nep]
        x_instru_im_jup = self.x_instru[mask_instru_im_jup].value ; y_instru_im_jup = self.y_instru[mask_instru_im_jup].value ; z_instru_im_jup = z_instru[mask_instru_im_jup]
        x_instru_tr_ter = self.x_instru[mask_instru_tr_ter].value ; y_instru_tr_ter = self.y_instru[mask_instru_tr_ter].value ; z_instru_tr_ter = z_instru[mask_instru_tr_ter]
        x_instru_ot_sol = self.x_instru[mask_instru_ot_sol].value ; y_instru_ot_sol = self.y_instru[mask_instru_ot_sol].value ; z_instru_ot_sol = z_instru[mask_instru_ot_sol]
        # PLOT 
        self.__plt.plot([], [], 'kv', ms=10, label=label_tr_ter) ; self.__plt.plot([], [], 'ko', ms=10, label=label_rv_nep) ; self.__plt.plot([], [], 'ks', ms=10, label=label_im_jup) ; self.__plt.plot([], [], 'kP', ms=10, label=label_ot_sol)
        self.__plt.plot(x_raw_tr_ter, y_raw_tr_ter, 'kv', alpha=0.5, ms=10) ; self.__plt.plot(x_raw_rv_nep, y_raw_rv_nep, 'ko', alpha=0.5, ms=10) ; self.__plt.plot(x_raw_im_jup, y_raw_im_jup, 'ks', alpha=0.5, ms=10) ; self.__plt.plot(x_raw_ot_sol, y_raw_ot_sol, 'kP', alpha=0.5, ms=10)
        self.__plt.scatter(x_instru_tr_ter, y_instru_tr_ter, s=100+100*z_instru_tr_ter/5, c=z_instru_tr_ter, ec="k", marker="v", cmap=self.cmap, norm=self.norm, zorder=3)
        self.__plt.scatter(x_instru_rv_nep, y_instru_rv_nep, s=100+100*z_instru_rv_nep/5, c=z_instru_rv_nep, ec="k", marker="o", cmap=self.cmap, norm=self.norm, zorder=3)
        self.__plt.scatter(x_instru_im_jup, y_instru_im_jup, s=100+100*z_instru_im_jup/5, c=z_instru_im_jup, ec="k", marker="s", cmap=self.cmap, norm=self.norm, zorder=3)
        self.__plt.scatter(x_instru_ot_sol, y_instru_ot_sol, s=100+100*z_instru_ot_sol/5, c=z_instru_ot_sol, ec="k", marker="P", cmap=self.cmap, norm=self.norm, zorder=3)
        if self.band == "INSTRU":
            self.lmin = self.config_data["lambda_range"]["lambda_min"] ; self.lmax = self.config_data["lambda_range"]["lambda_max"]
            self.R = 0.
            for band in self.config_data["gratings"]:
                self.R += self.config_data["gratings"][band].R/len(self.config_data["gratings"]) # mean resolution
        else:
            self.lmin = self.config_data['gratings'][self.band].lmin ; self.lmax = self.config_data['gratings'][self.band].lmax
            self.R = self.config_data["gratings"][self.band].R
        if self.systematics:
            if self.PCA:
                txt_syst = "(with systematics+PCA)"
            else:
                txt_syst = "(with systematics)"
        else:
            txt_syst = "(without systematics)"
        if self.instru == "NIRCam":
            self.__plt.set_title(f'Known exoplanets detection yield with {self.instru} ({self.spectrum_contributions} light with {self.model_planet})'+'\n on '+self.band+'-band (from '+str(round(self.lmin, 1))+' to '+str(round(self.lmax, 1))+f' µm) with '+'$t_{exp}$=' + str(round(self.exposure_time.get())) + 'mn ' + txt_syst, fontsize=18)
        else:
            self.__plt.set_title(f'Known exoplanets detection yield with {self.instru} ({self.spectrum_contributions} light with {self.model_planet})'+'\n on '+self.band+'-band (from '+str(round(self.lmin, 1))+' to '+str(round(self.lmax, 1))+f' µm with R ~ {int(round(self.R, -2))}) with '+'$t_{exp}$=' + str(round(self.exposure_time.get())) + 'mn ' + txt_syst, fontsize=18)
        if self.units == "Observational":
            self.__plt.set_xlabel(f'Maximum angular separation (in {x_raw.unit})', fontsize=18)
            self.__plt.set_ylabel(f'Planet-to-star contrast (on {self.band}-band)', fontsize=18)
            self.__plt.axvspan(iwa, owa, color='grey', alpha=0.5, lw=0, label="Working angle", zorder=2)
            if "thermal" in self.spectrum_contributions:
                self.__plt.set_ylim(1e-11, 1)
            else:
                self.__plt.set_ylim(1e-14, 1e-3)
            self.__plt.set_xlim(1e-2, 1e6)
        elif self.units == "Physical":
            self.__plt.set_xlabel(f'Semi Major Axis (in {x_raw.unit})', fontsize=18)
            self.__plt.set_ylabel(f"Planet mass (in {y_raw.unit})", fontsize=18)
            self.__plt.set_ylim(1e-1, 2.5e4) ; self.__plt.set_xlim(5e-3, 1e4)
        self.__plt.legend(loc="lower right", fontsize=16)
        self.__plt.text(.01, .99, f'total number of planets = {len(planet_table_raw)+len(self.planet_table)}', ha='left', va='top', transform=self.__plt.transAxes, fontsize=12)
        self.__plt.text(.01, .965, f'number of planets detected = {nb_detected} / {len(self.planet_table)}', ha='left', va='top', transform=self.__plt.transAxes, fontsize=12)
        if self.planet is not None:
            if self.planet["PlanetName"] in self.planet_table["PlanetName"]: 
                self.planet_index = get_planet_index(planet_table=self.planet_table, planet_name=self.planet["PlanetName"])
                self.planet = self.planet_table[self.planet_index]
                self.draw_table_parameters()
                self.__plt.plot(self.x_instru[self.planet_index], self.y_instru[self.planet_index], "kX", ms=14, zorder=4)
        self.__plt.set_yscale('log') ; self.__plt.set_xscale('log') ; self.__plt.grid(True, which='both', linestyle='--', linewidth=0.5) ; self.__plt.minorticks_on()
        self.__canvas.draw()
        
    def canvas_clicked(self, event):
        if event.xdata is not None and event.ydata is not None:
            c_x = floor(np.log10(event.xdata)) # see https://neutrium.net/general-engineering/accurate-readings-from-log-plots/
            c_y = floor(np.log10(event.ydata))
            a_clicked_x = np.log10(event.xdata/10**(c_x))
            a_clicked_y = np.log10(event.ydata/10**(c_y))
            a_data_x = np.log10(np.array(self.x_instru.value)/10**(c_x))
            a_data_y = np.log10(np.array(self.y_instru.value)/10**(c_y))
            delta_x = np.abs(a_clicked_x - a_data_x)
            delta_y = np.abs(a_clicked_y - a_data_y)
            self.planet_index = ((delta_x)**2 + (delta_y)**2).argmin()
            self.planet       = self.planet_table[self.planet_index]
            self.planet_name.set(self.planet["PlanetName"])
            self.draw_table_plot()
            
    def draw_table_parameters(self):
        self.__plt2.clear()
        self.__plt2.text(0.5, 1, f"{self.planet['PlanetName']} ({self.planet['PlanetType']})\n on {self.band}-band of {self.instru}", fontsize=22, weight='bold', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.9), va='center', ha='center', zorder=10)
        if self.band == "INSTRU":
            self.mag_planet =  self.planet["Planet"+self.band+"mag("+self.instru+")("+self.spectrum_contributions+")"]
            self.mag_star = self.planet["Star"+self.band+"mag("+self.instru+")"]
        else:
            self.mag_planet = self.planet["Planet"+self.band+"mag("+self.spectrum_contributions+")"]
            if self.band == "K":
                self.mag_star = float(self.planet["Star"+self.band+"mag"])
            else:
                self.mag_star = self.planet["Star"+self.band+"mag"]
        self.flux_ratio = 10**(-(self.mag_planet-self.mag_star)/2.5)
        data1 =    [[         'Planet', 'Star'], 
                    [ 'T', f'{int(round(float(self.planet["PlanetTeq"].value)))} {self.planet["PlanetTeq"].unit}', f'{int(round(float(self.planet["StarTeff"].value)))} {self.planet["StarTeff"].unit}'], 
                    [ 'lg', f'{round(float(self.planet["PlanetLogg"].value), 1)} {self.planet["PlanetLogg"].unit}', f'{round(float(self.planet["StarLogg"].value), 1)} {self.planet["StarLogg"].unit}'], 
                    [ 'M', f'{round(float(self.planet["PlanetMass"].value), 1)} {self.planet["PlanetMass"].unit}', f'{round(float(self.planet["StarMass"].value), 1)} {self.planet["StarMass"].unit}'], 
                    [ 'R', f'{round(float(self.planet["PlanetRadius"].value), 1)} {self.planet["PlanetRadius"].unit}', f'{round(float(self.planet["StarRadius"].value), 1)} {self.planet["StarRadius"].unit}'], 
                    [ 'mag', f'{round(self.mag_planet, 1)}', f'{round(self.mag_star, 1)}'], 
                    [ 'RV', f'{round(float(self.planet["PlanetRadialVelocity"].value), 1)} {self.planet["PlanetRadialVelocity"].unit}', f'{round(float(self.planet["StarRadialVelocity"].value), 1)} {self.planet["StarRadialVelocity"].unit}'], 
                    [ 'Vsini', f'{round(float(self.planet["PlanetVsini"].value), 1)} {self.planet["PlanetVsini"].unit}', f'{round(float(self.planet["StarVsini"].value), 1)} {self.planet["StarVsini"].unit}']]
        column_headers1 = data1.pop(0)
        row_headers1 = [row.pop(0) for row in data1]
        cell_text1 = [row for row in data1]
        row_colors1 = plt.cm.Oranges(np.full(len(row_headers1), 0.4))
        col_colors1 = plt.cm.Oranges(np.full(len(column_headers1), 0.4))
        table1 = self.__plt2.table(cellText=cell_text1, rowLabels=row_headers1, rowColours=row_colors1, rowLoc='center', colColours=col_colors1, colLabels=column_headers1, loc='center', bbox = [0.12, 0., 0.89, 0.4])
        table1.set_fontsize(16) 
        cells1 = table1.properties()["celld"]
        for (i, j), cell in cells1.items():
            cell.set_text_props(ha="center", va="center")
            cell.set_edgecolor('gray')
            cell.set_linewidth(0.5)
        data2 =  [[ 'S/N', f'{round(self.SNR[self.planet_index], 1)}'], 
                    [ 'Flux ratio', '{0:.1e}'.format(self.flux_ratio)], 
                    [ 'Max. sep.', f"{round(float(self.planet['AngSep'].value))} {self.planet['AngSep'].unit}"], 
                    [ 'SMA', f"{round(float(self.planet['SMA'].value), 1)} {self.planet['SMA'].unit}"], 
                    [ f'Discovery method', self.planet["DiscoveryMethod"]], 
                    [ f'Star spectral type', f"{self.planet['StarSpT']}"], 
                    [ f'Star age', f"{round(float(self.planet['StarAge'].value), 2)} {self.planet['StarAge'].unit}"], 
                    [ f'Inclination', f"{round(float(self.planet['Inc'].value))} °"], 
                    [ f'Distance', f"{round(float(self.planet['Distance'].value), 1)} {self.planet['Distance'].unit}"]]
        row_headers2 = [row.pop(0) for row in data2]
        cell_text2 = [row for row in data2]
        row_colors2 = plt.cm.Oranges(np.full(len(row_headers2), 0.4))
        table2 = self.__plt2.table(cellText=cell_text2, rowLabels=row_headers2, rowColours=row_colors2, rowLoc='center', loc='center', bbox = [0.49, 0.47, 0.52, 0.45])
        table2.set_fontsize(16) 
        cells2 = table2.properties()["celld"]
        for (i, j), cell in cells2.items():
            cell.set_text_props(ha="center", va="center")
            cell.set_edgecolor('gray')
            cell.set_linewidth(0.5)
        self.__plt2.text(1.05, -0.05, self.date, horizontalalignment='right', size=14, weight='light')
        
    def open_popup(self): # USELESS
        self.popup_state = "Open"
        self.popup = Toplevel(self)
        self.popup.title(self.planet["PlanetName"]) ; self.popup.attributes("-topmost", True)
        Button(self.popup, text="S/N", command=lambda *args:self.SNR_calculation(), bg="dark orange", fg="black", font=('Arial', 12, 'bold')).grid(row=11, column=0, sticky="nsew")
        Button(self.popup, text="Contrast", command=lambda *args:self.contrast_calculation(), bg="dark orange", fg="black", font=('Arial', 12, 'bold')).grid(row=11, column=2, sticky="nsew")
        self.popup.protocol('WM_DELETE_WINDOW', lambda: self.onclose())
    def onclose(self): # Func to be called when window is closing, passing the window name
        self.popup_state = "Close" # Set it to close
        self.popup.destroy() # Destroy the window

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
        planet_spectrum, planet_thermal, planet_reflected, star_spectrum = thermal_reflected_spectrum(self.planet, self.instru, thermal_model=self.thermal_model, reflected_model=self.reflected_model, wave_instru=None, wave_K=None, vega_spectrum_K=None, show=True)
        mag_p = self.planet["PlanetINSTRUmag("+self.instru+")("+self.spectrum_contributions+")"]
        mag_s = self.planet["StarINSTRUmag("+self.instru+")"]
        band0 = "instru"
        if self.calculation == "corner plot":
            if self.thermal_model == "None":
                planet_spectrum.model = "BT-Settl"
            else:
                planet_spectrum.model = self.thermal_model
            if self.band == "INSTRU": # picking the band with the highest S/N
                snr_max = 0 ; band_only = None
                for band in self.config_data["gratings"]:
                    snr = np.copy(np.sqrt(self.exposure_time.get()/self.planet['DIT_'+band]) * self.planet['signal_'+band] / np.sqrt( self.planet['sigma_fund_'+band]**2 + (self.exposure_time.get()/self.planet['DIT_'+band])*self.planet['sigma_syst_'+band]**2 ))
                    if snr > snr_max:
                        snr_max = snr ; band_only = band
            else:
                band_only = self.band
            try:
                FastCurves(instru=self.instru, band_only=band_only, calculation=self.calculation, T_planet=float(self.planet["PlanetTeq"].value), lg_planet=float(self.planet["PlanetLogg"].value), mag_star=mag_s, band0=band0, T_star=float(self.planet["StarTeff"].value), lg_star=float(self.planet["StarLogg"].value), exposure_time=self.exposure_time.get(), model_planet=planet_spectrum.model, mag_planet=mag_p, separation_planet=float(self.planet["AngSep"].value/1000), planet_name=self.planet["PlanetName"], systematic=self.systematics, PCA=self.PCA, show_plot=True, verbose=True, star_spectrum=star_spectrum, planet_spectrum=planet_spectrum, apodizer=self.apodizer, strehl=self.strehl)
            except Exception as e:
                print(f"The S/N ({round(self.SNR[self.planet_index])}) is sufficiently high for the precision of the parameter estimation to be likely limited by systematic effects rather than fundamental noises.")
        else:
            if self.band == "INSTRU":
                band_only = None 
            else:
                band_only = self.band
            FastCurves(instru=self.instru, band_only=band_only, calculation=self.calculation, T_planet=float(self.planet["PlanetTeq"].value), lg_planet=float(self.planet["PlanetLogg"].value), mag_star=mag_s, band0=band0, T_star=float(self.planet["StarTeff"].value), lg_star=float(self.planet["StarLogg"].value), exposure_time=self.exposure_time.get(), model_planet=self.model_planet, mag_planet=mag_p, separation_planet=float(self.planet["AngSep"].value/1000), planet_name=self.planet["PlanetName"], systematic=self.systematics, PCA=self.PCA, show_plot=True, verbose=True, star_spectrum=star_spectrum, planet_spectrum=planet_spectrum, apodizer=self.apodizer, strehl=self.strehl)

    def destroy_lower_buttons(self):
        try:
            self.button_band.destroy()
        except:
            pass
        try:
            self.button_calculation.destroy()
        except:
            pass
        try:
            self.button_model.destroy()
        except:
            pass
        try:
            self.button_systematics.destroy()
        except:
            pass
        try:
            self.button_visible_targets.destroy()
        except:
            pass



def is_planet_observable(target, times_utc, frame, is_night, min_elevation):
    target_altaz = target.transform_to(frame) # Calcul de l'altitude de la planète
    observable_times = (target_altaz.alt > min_elevation * u.deg) & is_night # Filtrage des moments où la planète est au-dessus de l'élévation minimale durant la nuit
    return np.any(observable_times) # Retourne True si la planète est détectable
def are_planets_observable(latitude, longitude, altitude, planet_table, date_obs, min_elevation):
    location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=altitude * u.m)
    tf = TimezoneFinder() # Obtention du fuseau horaire
    timezone_str = tf.timezone_at(lng=longitude, lat=latitude)
    local_tz = pytz.timezone(timezone_str)
    date_obs_local = datetime.strptime(date_obs, "%d/%m/%Y").astimezone(local_tz) # # Prépare le temps d'observation
    midnight = date_obs_local.replace(hour=0, minute=0, second=0)
    delta_midnight = np.linspace(-6, 6, 24) * u.hour  # 6 heures de chaque côté de minuit
    times_utc = Time(midnight) + delta_midnight
    frame = AltAz(obstime=times_utc, location=location)
    sun_altaz = get_body("sun", time=times_utc).transform_to(frame) # Calcul de l'altitude du Soleil
    is_night = sun_altaz.alt < 0 * u.deg
    planet_coords = [SkyCoord(ra=float(planet["RA"].value) * u.deg, dec=float(planet["Dec"].value) * u.deg) for planet in planet_table] # Créer une liste de SkyCoord pour chaque planète
    results = Parallel(n_jobs=-1)(delayed(is_planet_observable)(target, times_utc, frame, is_night, min_elevation) for target in planet_coords) # Parallélisation du calcul pour chaque planète
    return results



def FastYield_interface():
    app = MyWindow()
    app.mainloop()





















