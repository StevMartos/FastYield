import sys
from math import *
import tkinter as tk
from tkinter import *
from tkinter import ttk
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ttkwidgets.autocomplete import AutocompleteEntry

sys.path.insert(0, '../src')
from src.FastYield import *
from datetime import datetime

class MyWindow(tk.Tk): # https://koor.fr/Python/Tutoriel_Scipy_Stack/matplotlib_integration_ihm.wp

    def __init__(self):
        super().__init__()
        self.table = "Archive" # Archive / Simulated
        self.instru = "HARMONI"
        self.apodizer = "NO_SP"
        self.strehl = "JQ1"
        self.contrast = True
        self.exposure_time = DoubleVar(value=120)
        self.hmin = DoubleVar(value=30)
        self.thermal_model = "BT-Settl"
        self.reflected_model = "PICASO"
        self.model = self.thermal_model+"+"+self.reflected_model
        self.spectrum_contributions = "thermal+reflected"
        self.band = "INSTRU"
        self.only_visible_targets = False
        self.systematic = False
        self.t_instru = None
        self.popup = None
        self.popup_state = "Close"
        self.calculation = None
        self.planet = None
        self.name_planet = StringVar()
        
        self.date = datetime.now().strftime("%d/%m/%Y")
        
        self.title("FastYield")
        try:
            self.state('zoomed') #works fine on Windows!
        except:
            m = self.maxsize()
            self.geometry('{}x{}+0+0'.format(*m))    
        self.configure(bg='black')
        
        Label(self, text="FastYield", font='Magneto 30 bold',bg="black",fg="dark orange").pack()
        
        # CHOIX DE LA TABLE
        self.button_table = tk.Frame(self) ; self.button_table.pack(side=tk.TOP, fill=tk.X)
        self.__btn_archive = tk.Button(self.button_table, text="ARCHIVE TABLE", command=self.btn_archive_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_archive.grid(row=0, column=0, sticky="nsew") ; self.button_table.grid_columnconfigure(0, weight=1)
        self.__btn_simulated = tk.Button(self.button_table, text="SIMULATED TABLE", command=self.btn_simulated_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_simulated.grid(row=0, column=1, sticky="nsew") ; self.button_table.grid_columnconfigure(1, weight=1)

        # CHOIX DE L'INSTRUMENT 
        self.button_instru = tk.Frame(self) ; self.button_instru.pack(side=tk.TOP, fill=tk.X)
        self.__btn_harmoni = tk.Button(self.button_instru, text="ELT/HARMONI", command=self.btn_harmoni_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_harmoni.grid(row=0, column=0, sticky="nsew") ; self.button_instru.grid_columnconfigure(0, weight=1)
        self.__btn_andes = tk.Button(self.button_instru, text="ELT/ANDES", command=self.btn_andes_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_andes.grid(row=0, column=1, sticky="nsew") ; self.button_instru.grid_columnconfigure(1, weight=1)
        self.__btn_eris = tk.Button(self.button_instru, text="VLT/ERIS", command=self.btn_eris_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_eris.grid(row=0, column=2, sticky="nsew") ; self.button_instru.grid_columnconfigure(2, weight=1)
        self.__btn_mirimrs = tk.Button(self.button_instru, text="JWST/MIRI/MRS", command=self.btn_mirimrs_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_mirimrs.grid(row=0, column=3, sticky="nsew") ; self.button_instru.grid_columnconfigure(3, weight=1)
        self.__btn_nircam = tk.Button(self.button_instru, text="JWST/NIRSpec/IFU", command=self.btn_nirspec_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_nircam.grid(row=0, column=4, sticky="nsew") ; self.button_instru.grid_columnconfigure(4, weight=1)
        self.__btn_nircam = tk.Button(self.button_instru, text="JWST/NIRCam", command=self.btn_nircam_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_nircam.grid(row=0, column=5, sticky="nsew") ; self.button_instru.grid_columnconfigure(5, weight=1)
        
        # CHOIX DES UNITES + texp  + name
        self.button_units_texp_name = tk.Frame(self) ; self.button_units_texp_name.pack(side=tk.TOP, fill=tk.X)
        self.__btn_physic = tk.Button(self.button_units_texp_name, text="Physical units", command=self.btn_physic_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_physic.grid(row=0, column=0, sticky="nsew") ; self.button_units_texp_name.grid_columnconfigure(0, weight=1)
        self.__btn_contrast = tk.Button(self.button_units_texp_name, text="Observational units", command=self.btn_contrast_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_contrast.grid(row=0, column=3, sticky="nsew") ; self.button_units_texp_name.grid_columnconfigure(3, weight=1)
        self.button_texp = tk.LabelFrame(self.button_units_texp_name) ; self.button_texp.grid(row=0, column=1, sticky="nsew") ; self.button_texp.grid_columnconfigure(1, weight=1)
        self.__btn_exp = Label(self.button_texp, text = "Exposure time (in mn) :",fg="black",font=('Arial', 12, 'bold')) ; self.__btn_exp.grid(column=0, row=0)
        self.__btn_exp = Button(self.button_texp, text = "Enter",width=10,height=1, command=self.draw_table_plot,bg="dark orange",fg="black",font=('Arial', 12, 'bold')) ; self.__btn_exp.grid(column=2, row=0) 
        self.__btn_exp_entry = Entry(self.button_texp, width=10,textvariable=self.exposure_time,justify=CENTER,font=('Arial', 12, 'bold')) ; self.__btn_exp_entry.grid(column=1, row=0)
        self.__btn_exp_entry.bind("<Return>", lambda _ : self.draw_table_plot())
        
        # On instancie le Canvas MPL.
        self.__fig = plt.figure(constrained_layout=True) ; self.__canvas = FigureCanvasTkAgg(self.__fig, master=self) ; self.__canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1) ; self.__canvas.mpl_connect("button_press_event", self.canvas_clicked)
        self.__plt = self.__fig.add_subplot(1,4,(1,3)) ; self.__plt.tick_params(axis='both', labelsize=16)
        self.cmap = plt.get_cmap("rainbow") ; self.norm = plt.Normalize(0.,5.) ; sm =  ScalarMappable(norm=self.norm, cmap=self.cmap) ; sm.set_array([])
        self.__cbar = self.__fig.colorbar(sm,ax=self.__plt,pad=0.025,shrink=0.8) ; self.__cbar.set_label('S/N', fontsize=20, labelpad=20, rotation=270) ; self.__cbar.ax.tick_params(labelsize=16)
        self.__fig.set_constrained_layout_pads(w_pad=0.4, h_pad=0.4, wspace=0., hspace=0.)
        self.__plt2 = self.__fig.add_subplot(1,4,4)
        self.__plt2.set_frame_on(False)
        self.__plt2.tick_params(left=False,right=False,labelleft=False,labelbottom=False,bottom=False) 
        
        # On initie une table par défaut
        self.t_instru = load_planet_table(self.table+"_Pull_"+self.instru+"_"+self.apodizer+"_"+self.strehl+"_without_systematics_"+self.model+".ecsv")
    
        self.button_name = tk.LabelFrame(self.button_units_texp_name) ; self.button_name.grid(row=0, column=2, sticky="nsew") ; self.button_name.grid_columnconfigure(2, weight=1)
        self.__btn_name = Label(self.button_name, text = "Planet name :",fg="black",font=('Arial', 12, 'bold')) ; self.__btn_name.grid(column=0, row=0)
        self.__btn_name = Button(self.button_name, text = "Enter",width=10,height=1, command=self.enter_name_planet,bg="dark orange",fg="black",font=('Arial', 12, 'bold')) ; self.__btn_name.grid(column=2, row=0) 
        self.__btn_name_entry = AutocompleteEntry(self.button_name, width=20,font=('Arial', 12, 'bold'),completevalues=list(self.t_instru["PlanetName"]),textvariable=self.name_planet,justify=CENTER) ; self.__btn_name_entry.grid(column=1, row=0)
        self.__btn_name_entry.bind("<Return>", lambda _ : self.enter_name_planet())
    
        # On initie le plot
        self.btn_table_clicked()
        
    def enter_name_planet(self):
        if self.name_planet.get() in self.t_instru["PlanetName"]:
            self.planet_index = planet_index(planet_table=self.t_instru,planet_name=self.name_planet.get())
            self.planet = self.t_instru[self.planet_index]
        else :
            self.planet = None
        self.draw_table_plot()
            
    def get_latitude(self):
        if self.instru == "HARMONI" or self.instru=="ANDES":
            self.latitude = -24.589317 # °N
        elif self.instru == "ERIS":
            self.latitude = -24.627167 # °N
            
    def btn_archive_clicked(self):
        self.destroy_lower_buttons()
        self.table = "Archive"
        self.btn_table_clicked()
    def btn_simulated_clicked(self):
        self.destroy_lower_buttons()
        self.table = "Simulated"
        self.btn_table_clicked()
    def btn_table_clicked(self):
        if self.instru == "HARMONI":
            self.btn_harmoni_clicked()
        elif self.instru == "ANDES":
            self.btn_andes_clicked()
        elif self.instru == "ERIS":
            self.btn_eris_clicked()
        elif self.instru == "MIRIMRS":
            self.btn_mirimrs_clicked()
        elif self.instru == "NIRSpec":
            self.btn_mirimrs_clicked()
        elif self.instru == "NIRCam":
            self.btn_nircam_clicked()
        else :
            raise KeyError("PLEASE ADD THE INSTRUMENT IN THE FUNCTION !")
            
    def btn_harmoni_clicked(self):
        self.destroy_lower_buttons()
        self.instru = "HARMONI"
        self.config_data = get_config_data(self.instru)
        self.apodizer = "NO_SP"
        self.strehl = "JQ1"
        self.systematic = False
        self.thermal_model = "BT-Settl"
        self.reflected_model = "PICASO"
        self.model = self.thermal_model+"+"+self.reflected_model
        self.spectrum_contributions = "thermal+reflected"
        self.band = "INSTRU"
        self.create_button_band_calculation()
        if self.table == "Archive":
            self.create_button_model()
            self.create_button_visisble_targets()
        self.draw_table_plot()
    def btn_andes_clicked(self):
        self.destroy_lower_buttons()
        self.instru = "ANDES"
        self.config_data = get_config_data(self.instru)
        self.apodizer = "NO_SP"
        self.strehl = "JQ1"
        self.systematic = False
        self.thermal_model = "BT-Settl"
        self.reflected_model = "PICASO"
        self.model = self.thermal_model+"+"+self.reflected_model
        self.spectrum_contributions = "thermal+reflected"
        self.band = "INSTRU"
        self.create_button_band_calculation()
        if self.table == "Archive":
            self.create_button_model()
            self.create_button_visisble_targets()
        self.draw_table_plot()
    def btn_eris_clicked(self):
        self.destroy_lower_buttons()
        self.instru = "ERIS"
        self.config_data = get_config_data(self.instru)
        self.apodizer = "NO_SP"
        self.strehl = "JQ0"
        self.systematic = False
        self.thermal_model = "BT-Settl"
        self.reflected_model = "PICASO"
        self.model = self.thermal_model+"+"+self.reflected_model
        self.spectrum_contributions = "thermal+reflected"
        self.band = "INSTRU"
        self.create_button_band_calculation()
        if self.table == "Archive":
            self.create_button_model()
            self.create_button_visisble_targets()
        self.draw_table_plot()
    def btn_mirimrs_clicked(self):
        self.destroy_lower_buttons()
        self.instru = "MIRIMRS"
        self.config_data = get_config_data(self.instru)
        self.apodizer = "NO_SP"
        self.strehl = "NO_JQ"
        self.model = "BT-Settl"
        self.thermal_model = "BT-Settl"
        self.reflected_model = "None"
        self.spectrum_contributions = "thermal"
        self.band = "INSTRU"
        self.create_button_band_calculation()
        if self.table == "Archive":
            self.create_button_model()
        self.create_button_systematics()
        self.only_visible_targets = False
        self.draw_table_plot()
    def btn_nirspec_clicked(self):
        self.destroy_lower_buttons()
        self.instru = "NIRSpec"
        self.config_data = get_config_data(self.instru)
        self.apodizer = "NO_SP"
        self.strehl = "NO_JQ"
        self.systematic = False
        self.thermal_model = "BT-Settl"
        self.reflected_model = "PICASO"
        self.model = self.thermal_model+"+"+self.reflected_model
        self.spectrum_contributions = "thermal+reflected"
        self.band = "INSTRU"
        self.create_button_band_calculation()
        if self.table == "Archive":
            self.create_button_model()
        self.create_button_systematics()
        self.only_visible_targets = False
        self.draw_table_plot()
    def btn_nircam_clicked(self):
        self.destroy_lower_buttons()
        self.instru = "NIRCam"
        self.config_data = get_config_data(self.instru)
        self.apodizer = "NO_SP"
        self.strehl = "NO_JQ"
        self.systematic = False
        self.thermal_model = "BT-Settl"
        self.reflected_model = "PICASO"
        self.model = self.thermal_model+"+"+self.reflected_model
        self.spectrum_contributions = "thermal+reflected"
        self.band = "INSTRU"
        self.create_button_band_calculation()
        if self.table == "Archive":
            self.create_button_model()
        self.only_visible_targets = False
        self.draw_table_plot()
    
    def create_button_systematics(self):
        self.button_systematics = tk.Frame(self) ; self.button_systematics.pack(side=tk.TOP, fill=tk.X)
        self.__btn = tk.Button(self.button_systematics, text="Without systematics", command=self.btn_no_systematics_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=0, sticky="nsew") ; self.button_systematics.grid_columnconfigure(0, weight=1)
        self.__btn = tk.Button(self.button_systematics, text="With systematics", command=self.btn_systematic_limit_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=1, sticky="nsew") ; self.button_systematics.grid_columnconfigure(1, weight=1)
        
    def btn_no_systematics_clicked(self):
        self.systematic = False
        self.draw_table_plot()
    def btn_systematic_limit_clicked(self):
        self.systematic = True
        self.draw_table_plot()
    
    def create_button_band_calculation(self):
        list_band = ["INSTRU"]
        for name_band in self.config_data['gratings'] :
            list_band.append(name_band)
        self.button_band_calculation = tk.Frame(self) ; self.button_band_calculation.pack(side=tk.TOP, fill=tk.X)
        self.button_bands = tk.LabelFrame(self.button_band_calculation,bg="dark orange") ; self.button_bands.grid(row=0, column=0, sticky="nsew") ; self.button_band_calculation.grid_columnconfigure(0, weight=1)
        self.__txt_bands = Label(self.button_bands, text = "Bandwidth : ",fg="black",bg="dark orange",font=('Arial', 14, 'bold')) ; self.__txt_bands.grid(column=0, row=0) ; self.button_bands.grid_columnconfigure(0, weight=1)
        self.__list_bands = ttk.Combobox(self.button_bands,state='readonly',font=('Arial', 14,'bold'),justify='center') ; self.__list_bands.grid(column=1, row=0) ; self.button_bands.grid_columnconfigure(1, weight=1)
        popdown = self.__list_bands.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_bands)
        self.__list_bands.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_bands['font'])
        self.__list_bands['values'] = list_band
        self.__list_bands.current(0) #index de l'élément sélectionné
        self.__list_bands.bind("<<ComboboxSelected>>", lambda _ : self.band_clicked())
        self.SNR_calculation_button = Button(self.button_band_calculation, text="S/N curves", command=lambda *args:self.SNR_calculation(),bg="dark orange",fg="black",font=('Arial', 12, 'bold')) ; self.SNR_calculation_button.grid(row=0, column=1, sticky="nsew")  ; self.button_band_calculation.grid_columnconfigure(1, weight=1)
        self.contrast_calculation_button = Button(self.button_band_calculation, text="Contrast curves", command=lambda *args:self.contrast_calculation(),bg="dark orange",fg="black",font=('Arial', 12, 'bold')) ; self.contrast_calculation_button.grid(row=0, column=2, sticky="nsew")  ; self.button_band_calculation.grid_columnconfigure(2, weight=1)
    def band_clicked(self):
        self.band = self.__list_bands.get()
        self.draw_table_plot()

    def create_button_model(self):
        self.button_model = tk.Frame(self) ; self.button_model.pack(side=tk.TOP, fill=tk.X)
        self.button_thermal_model = tk.LabelFrame(self.button_model) ; self.button_thermal_model.grid(row=0, column=0, sticky="nsew") ; self.button_model.grid_columnconfigure(0, weight=1)
        self.__txt_thermal_model = Label(self.button_thermal_model, text = "Thermal contribution (model) : ",fg="black",font=('Arial', 14, 'bold')) ; self.__txt_thermal_model.grid(column=0, row=0) ; self.button_thermal_model.grid_columnconfigure(0, weight=1)
        self.__list_thermal_model = ttk.Combobox(self.button_thermal_model,state='readonly',font=('Arial', 14,'bold'),justify='center') ; self.__list_thermal_model.grid(column=1, row=0) ; self.button_thermal_model.grid_columnconfigure(1, weight=1)
        popdown = self.__list_thermal_model.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_thermal_model)
        self.__list_thermal_model.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_thermal_model['font'])
        if self.config_data["lambda_range"]["lambda_max"] < 6 :
            self.__list_thermal_model['values'] = ("None","BT-Settl","Exo-REM","PICASO")
        else :
            self.__list_thermal_model['values'] = ("None","BT-Settl","Exo-REM")
        self.__list_thermal_model.current(1) #index de l'élément sélectionné
        self.__list_thermal_model.bind("<<ComboboxSelected>>", lambda _ : self.contribution_clicked())
        if self.config_data["lambda_range"]["lambda_max"] < 6 :
            self.button_reflected_model = tk.LabelFrame(self.button_model) ; self.button_reflected_model.grid(row=0, column=1, sticky="nsew") ; self.button_model.grid_columnconfigure(1, weight=1)
            self.__txt_reflected_model = Label(self.button_reflected_model, text = "Reflected contribution (albedo model) : ",fg="black",font=('Arial', 14, 'bold')) ; self.__txt_reflected_model.grid(column=0, row=0) ; self.button_reflected_model.grid_columnconfigure(0, weight=1)
            self.__list_reflected_model = ttk.Combobox(self.button_reflected_model,state='readonly',font=('Arial', 14,'bold'),justify='center') ; self.__list_reflected_model.grid(column=1, row=0) ; self.button_reflected_model.grid_columnconfigure(1, weight=1)
            popdown = self.__list_reflected_model.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_reflected_model)
            self.__list_reflected_model.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_reflected_model['font'])
            self.__list_reflected_model['values'] = ("None","PICASO","tellurics","flat")
            self.__list_reflected_model.current(1) #index de l'élément sélectionné
            self.__list_reflected_model.bind("<<ComboboxSelected>>", lambda _ : self.contribution_clicked())
    def contribution_clicked(self):
        try :
            self.thermal_model = self.__list_thermal_model.get()
        except :
            pass
        try :
            self.reflected_model = self.__list_reflected_model.get()
        except :
            pass
        if self.thermal_model != "None" and self.reflected_model != "None" :
            self.spectrum_contributions = "thermal+reflected"
            self.model = self.thermal_model+"+"+self.reflected_model
        elif self.thermal_model != "None" and self.reflected_model == "None" :
            self.spectrum_contributions = "thermal"
            if self.thermal_model == "PICASO":
                self.model = self.thermal_model+"_thermal_only"
            else :
                self.model = self.thermal_model
        elif self.thermal_model == "None" and self.reflected_model != "None" :
            self.spectrum_contributions = "reflected"
            if self.reflected_model == "PICASO":
                self.model = self.reflected_model+"_reflected_only"
            else :
                self.model = self.reflected_model
        self.draw_table_plot()
        
    def create_button_visisble_targets(self):
        # CHOIX DES UNITES + texp
        self.button_visible_targets = tk.Frame(self) ; self.button_visible_targets.pack(side=tk.TOP, fill=tk.X)
        self.__btn = tk.Button(self.button_visible_targets, text="All targets", command=self.btn_all_targets_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=0, sticky="nsew") ; self.button_visible_targets.grid_columnconfigure(0, weight=1)
        self.__btn = tk.Button(self.button_visible_targets, text="Only visible targets from the observation site", command=self.btn_visible_targets_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=1, sticky="nsew") ; self.button_visible_targets.grid_columnconfigure(1, weight=1)
        self.button_hmin = tk.LabelFrame(self.button_visible_targets) ; self.button_hmin.grid(row=0, column=2, sticky="nsew") ; self.button_hmin.grid_columnconfigure(2, weight=1)
        self.__txt_hmin = Label(self.button_hmin, text = "Minimum elevation (in °) : ",fg="black",font=('Arial', 12, 'bold')) ; self.__txt_hmin.grid(column=0, row=0)
        self.__btn_hmin = Button(self.button_hmin, text = "Enter",width=10,height=1, command=self.draw_table_plot,bg="dark orange",fg="black",font=('Arial', 12, 'bold')) ; self.__btn_hmin.grid(column=2, row=0) 
        self.__btn_entry_hmin = Entry(self.button_hmin, width=10,textvariable=self.hmin,justify=CENTER,font=('Arial', 12, 'bold')) ; self.__btn_entry_hmin.grid(column=1, row=0) 
        self.__btn_entry_hmin.bind("<Return>", lambda _ : self.draw_table_plot())
    def btn_all_targets_clicked(self):
        self.only_visible_targets = False
        self.draw_table_plot()
    def btn_visible_targets_clicked(self):
        self.only_visible_targets = True
        self.draw_table_plot()

    def btn_physic_clicked(self):
        self.contrast = False
        self.draw_table_plot()
    def btn_contrast_clicked(self):
        self.contrast = True
        self.draw_table_plot()
        
    def draw_table_plot(self):
        if self.popup is not None :
            self.popup.destroy()
        self.__plt.clear()
        try :
            t_exp = self.exposure_time.get()
        except :
            t_exp = 120
        # chargement des tables de planètes
        t_raw = load_planet_table(self.table+"_Pull_for_FastCurves.ecsv")
        if self.table == "Archive":
            t_raw = t_raw[np.logical_not(t_raw['AngSep'].mask)]
        if self.systematic :
            self.t_instru = load_planet_table(self.table+"_Pull_"+self.instru+"_"+self.apodizer+"_"+self.strehl+"_with_systematics_"+self.model+".ecsv")
        else :
            self.t_instru = load_planet_table(self.table+"_Pull_"+self.instru+"_"+self.apodizer+"_"+self.strehl+"_without_systematics_"+self.model+".ecsv")
        # Filtrage les planètes non-visibles depuis le site d'observation
        if self.only_visible_targets : # on filtre les planètes non-visibles depuis le site d'observation
            self.get_latitude()
            t_raw = t_raw[(90-np.abs(self.latitude-t_raw["Dec"].value))>self.hmin.get()]
            self.t_instru = self.t_instru[(90-np.abs(self.latitude-self.t_instru["Dec"].value))>self.hmin.get()]
        # calcul du SNR de chaque planète sur la bande
        self.SNR = np.copy(np.sqrt(t_exp/self.t_instru['DIT_'+self.band]) * self.t_instru['signal_'+self.band] / np.sqrt(  self.t_instru['sigma_non-syst_'+self.band]**2 + (t_exp/self.t_instru['DIT_'+self.band])*self.t_instru['sigma_syst_'+self.band]**2 ))
        z_instru = np.copy(self.SNR) ; z_instru[z_instru>=5] = 5 ; nb_detected = len(self.SNR[self.SNR>=5])
        # WORKING ANGLE
        if self.instru == "HARMONI":
            iwa = self.config_data["apodizers"][self.apodizer].sep
        else :
            lambda_c = (self.config_data["lambda_range"]["lambda_min"]+self.config_data["lambda_range"]["lambda_max"])/2 *1e-6*u.m
            diameter = self.config_data['telescope']['diameter'] *u.m
            iwa = lambda_c/diameter*u.rad ; iwa = iwa.to(u.mas) ; iwa = iwa.value
        owa = self.config_data["spec"]["FOV"]*1000/2 # en mas
        t_raw = t_raw[t_raw["AngSep"] <= iwa * u.mas]
        # Definition des vecteurs x, y et z
        if self.contrast :
            x_raw = t_raw["AngSep"] ; self.x_instru = self.t_instru["AngSep"]
            if self.band != "INSTRU":
                y_raw = 10**(-(t_raw["Planet"+self.band+"mag("+self.spectrum_contributions+")"]-np.array(t_raw["Star"+self.band+"mag"]))/2.5)*u.dimensionless_unscaled
                self.y_instru  = 10**(-(self.t_instru["Planet"+self.band+"mag("+self.spectrum_contributions+")"]-np.array(self.t_instru["Star"+self.band+"mag"]))/2.5)*u.dimensionless_unscaled
            else :
                y_raw = 10**(-(t_raw["Planet"+self.band+"mag("+self.instru+")("+self.spectrum_contributions+")"]-np.array(t_raw["Star"+self.band+"mag("+self.instru+")"]))/2.5)*u.dimensionless_unscaled
                self.y_instru  = 10**(-(self.t_instru["Planet"+self.band+"mag("+self.instru+")("+self.spectrum_contributions+")"]-np.array(self.t_instru["Star"+self.band+"mag("+self.instru+")"]))/2.5)*u.dimensionless_unscaled
        else :
            x_raw = t_raw['SMA'] ; self.x_instru = self.t_instru['SMA']
            y_raw = t_raw["PlanetMass"] ; self.y_instru = self.t_instru["PlanetMass"]
        if self.table == "Archive":
            mask_raw_rv_nep = t_raw["DiscoveryMethod"]=="Radial Velocity" ; mask_instru_rv_nep =  self.t_instru["DiscoveryMethod"]=="Radial Velocity" ; label_rv_nep = "Radial Velocity"
            mask_raw_im_jup = t_raw["DiscoveryMethod"]=="Imaging" ; mask_instru_im_jup =  self.t_instru["DiscoveryMethod"]=="Imaging" ; label_im_jup = "Direct Imaging"
            mask_raw_tr_ter = t_raw["DiscoveryMethod"]=="Transit" ; mask_instru_tr_ter =  self.t_instru["DiscoveryMethod"]=="Transit" ; label_tr_ter = "Transit"
            mask_raw_ot_sol = (t_raw["DiscoveryMethod"]!="Imaging") & (t_raw["DiscoveryMethod"]!="Radial Velocity") & (t_raw["DiscoveryMethod"]!="Transit") ; mask_instru_ot_sol =  (self.t_instru["DiscoveryMethod"]!="Imaging") & (self.t_instru["DiscoveryMethod"]!="Radial Velocity") & (self.t_instru["DiscoveryMethod"]!="Transit") ; label_ot_sol = "Other"
        elif self.table == "Simulated":
            mask_raw_rv_nep = t_raw["PlanetType"]=="Neptunian" ; mask_instru_rv_nep =  self.t_instru["PlanetType"]=="Neptunian" ; label_rv_nep = "Neptunian worlds"
            mask_raw_im_jup = t_raw["PlanetType"]=="Jovian" ; mask_instru_im_jup =  self.t_instru["PlanetType"]=="Jovian" ; label_im_jup = "Jovian worlds"
            mask_raw_tr_ter = t_raw["PlanetType"]=="Terran" ; mask_instru_tr_ter =  self.t_instru["PlanetType"]=="Terran" ; label_tr_ter = "Terran worlds"
            mask_raw_ot_sol = t_raw["PlanetType"]=="Stellar" ; mask_instru_ot_sol =  self.t_instru["PlanetType"]=="Stellar" ; label_ot_sol = "Stellar worlds"
        x_raw_rv_nep = x_raw[mask_raw_rv_nep].value ; y_raw_rv_nep = y_raw[mask_raw_rv_nep].value
        x_raw_im_jup = x_raw[mask_raw_im_jup].value ; y_raw_im_jup = y_raw[mask_raw_im_jup].value
        x_raw_tr_ter = x_raw[mask_raw_tr_ter].value ; y_raw_tr_ter = y_raw[mask_raw_tr_ter].value
        x_raw_ot_sol = x_raw[mask_raw_ot_sol].value ; y_raw_ot_sol = y_raw[mask_raw_ot_sol].value
        x_instru_rv_nep = self.x_instru[mask_instru_rv_nep].value ; y_instru_rv_nep = self.y_instru[mask_instru_rv_nep].value ; z_instru_rv_nep = z_instru[mask_instru_rv_nep]
        x_instru_im_jup = self.x_instru[mask_instru_im_jup].value ; y_instru_im_jup = self.y_instru[mask_instru_im_jup].value ; z_instru_im_jup = z_instru[mask_instru_im_jup]
        x_instru_tr_ter = self.x_instru[mask_instru_tr_ter].value ; y_instru_tr_ter = self.y_instru[mask_instru_tr_ter].value ; z_instru_tr_ter = z_instru[mask_instru_tr_ter]
        x_instru_ot_sol = self.x_instru[mask_instru_ot_sol].value ; y_instru_ot_sol = self.y_instru[mask_instru_ot_sol].value ; z_instru_ot_sol = z_instru[mask_instru_ot_sol]
        # PLOT 
        self.__plt.plot([],[],'kv',ms=10,label=label_tr_ter) ; self.__plt.plot([],[],'ko',ms=10,label=label_rv_nep) ; self.__plt.plot([],[],'ks',ms=10,label=label_im_jup) ; self.__plt.plot([],[],'kP',ms=10,label=label_ot_sol)
        self.__plt.plot(x_raw_tr_ter,y_raw_tr_ter,'kv',alpha=0.5,ms=10) ; self.__plt.plot(x_raw_rv_nep,y_raw_rv_nep,'ko',alpha=0.5,ms=10) ; self.__plt.plot(x_raw_im_jup,y_raw_im_jup,'ks',alpha=0.5,ms=10) ; self.__plt.plot(x_raw_ot_sol,y_raw_ot_sol,'kP',alpha=0.5,ms=10)
        self.__plt.scatter(x_instru_tr_ter, y_instru_tr_ter, s=100+100*z_instru_tr_ter/5, c=z_instru_tr_ter, ec="k",marker="v",cmap=self.cmap,norm=self.norm,zorder=3)
        self.__plt.scatter(x_instru_rv_nep, y_instru_rv_nep,s=100+100*z_instru_rv_nep/5, c=z_instru_rv_nep, ec="k", marker="o",cmap=self.cmap,norm=self.norm,zorder=3)
        self.__plt.scatter(x_instru_im_jup, y_instru_im_jup,s=100+100*z_instru_im_jup/5, c=z_instru_im_jup, ec="k",marker="s",cmap=self.cmap,norm=self.norm,zorder=3)
        self.__plt.scatter(x_instru_ot_sol, y_instru_ot_sol,s=100+100*z_instru_ot_sol/5, c=z_instru_ot_sol, ec="k",marker="P",cmap=self.cmap,norm=self.norm,zorder=3)
        if self.band == "INSTRU":
            self.lmin = self.config_data["lambda_range"]["lambda_min"] ; self.lmax = self.config_data["lambda_range"]["lambda_max"]
            self.R = 0.
            for band in self.config_data["gratings"] :
                self.R += self.config_data["gratings"][band].R/len(self.config_data["gratings"])
        else :
            self.lmin = self.config_data['gratings'][self.band].lmin ; self.lmax = self.config_data['gratings'][self.band].lmax
            self.R = self.config_data["gratings"][self.band].R
        if self.systematic:
            txt_syst = "(with systematics)"
        else :
            txt_syst = "(without systematics)"
        if self.instru=="NIRCam":
            self.__plt.set_title(f'Known exoplanets detection yield with {self.instru} ({self.spectrum_contributions} light with {self.model})'+'\n on '+self.band+'-band (from '+str(round(self.lmin,1))+' to '+str(round(self.lmax,1))+f' µm) with '+'$t_{exp}$=' + str(round(t_exp)) + 'mn ' + txt_syst, fontsize = 24)
        else :
            self.__plt.set_title(f'Known exoplanets detection yield with {self.instru} ({self.spectrum_contributions} light with {self.model})'+'\n on '+self.band+'-band (from '+str(round(self.lmin,1))+' to '+str(round(self.lmax,1))+f' µm with R ~ {int(round(self.R,-2))}) with '+'$t_{exp}$=' + str(round(t_exp)) + 'mn ' + txt_syst, fontsize = 24)
        if self.contrast :
            self.__plt.set_xlabel(f'Angular separation (in {x_raw.unit})', fontsize = 20)
            self.__plt.set_ylabel(f'Contrast (on {self.band}-band)', fontsize = 20)
            self.__plt.axvspan(iwa, owa, color='k', alpha=0.5, lw=0,label="Working angle",zorder=2)
            if "thermal" in self.spectrum_contributions :
                self.__plt.set_ylim(1e-11,1)
            else :
                self.__plt.set_ylim(1e-14,1e-3)
            self.__plt.set_xlim(1e-2,1e6)
        else :
            self.__plt.set_xlabel(f'Semi Major Axis (in {x_raw.unit})', fontsize = 20)
            self.__plt.set_ylabel(f"Planet mass (in {y_raw.unit})", fontsize = 20)
            self.__plt.set_ylim(1e-1,2.5e4) ; self.__plt.set_xlim(5e-3,1e4)
        self.__plt.legend(loc="lower right",fontsize=16)
        self.__plt.text(.01, .99, f'total number of planets = {len(t_raw)+len(self.t_instru)}', ha='left', va='top', transform=self.__plt.transAxes,fontsize=12)
        self.__plt.text(.01, .965, f'number of planets detected = {nb_detected} / {len(self.t_instru)}', ha='left', va='top', transform=self.__plt.transAxes,fontsize=12)
        if self.planet is not None :
            if self.planet["PlanetName"] in self.t_instru["PlanetName"] : 
                self.planet_index = planet_index(planet_table=self.t_instru,planet_name=self.planet["PlanetName"])
                self.planet = self.t_instru[self.planet_index]
                self.draw_table_parameters()
                self.__plt.plot(self.x_instru[self.planet_index], self.y_instru[self.planet_index],"kX",ms=14,zorder=4)
        self.__plt.set_yscale('log') ; self.__plt.set_xscale('log') ; self.__plt.grid(True)
        self.__canvas.draw()
        
    def canvas_clicked(self,event):
        if event.xdata is not None and event.ydata is not None :
            c_x = floor(np.log10(event.xdata)) # see https://neutrium.net/general-engineering/accurate-readings-from-log-plots/
            c_y = floor(np.log10(event.ydata))
            a_clicked_x = np.log10(event.xdata/10**(c_x))
            a_clicked_y = np.log10(event.ydata/10**(c_y))
            a_data_x = np.log10(np.array(self.x_instru.value)/10**(c_x))
            a_data_y = np.log10(np.array(self.y_instru.value)/10**(c_y))
            delta_x = np.abs(a_clicked_x - a_data_x)
            delta_y = np.abs(a_clicked_y - a_data_y)
            self.planet_index = ((delta_x)**2+(delta_y)**2).argmin()
            self.planet = self.t_instru[self.planet_index]
            self.name_planet.set(self.planet["PlanetName"])
            self.draw_table_plot()
            
    def draw_table_parameters(self):
        self.__plt2.clear()
        self.__plt2.text(0.5,1, self.planet["PlanetName"]+f"\n on {self.band}-band of {self.instru}",fontsize=20,weight='bold',bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'none', 'pad': 1}, verticalalignment='center', horizontalalignment='center',zorder=10)
        if self.band == "INSTRU":
            self.mag_planet =  self.planet["Planet"+self.band+"mag("+self.instru+")("+self.spectrum_contributions+")"]
            self.mag_star = self.planet["Star"+self.band+"mag("+self.instru+")"]
        else :
            self.mag_planet = self.planet["Planet"+self.band+"mag("+self.spectrum_contributions+")"]
            if self.band == "K":
                self.mag_star = float(self.planet["Star"+self.band+"mag"])
            else :
                self.mag_star = self.planet["Star"+self.band+"mag"]
        self.flux_ratio = 10**(-(self.mag_planet-self.mag_star)/2.5)
        self.data =  [[         'Planet', 'Star'],
                    [ 'T',  f'{int(round(float(self.planet["PlanetTeq"].value)))} {self.planet["PlanetTeq"].unit}', f'{int(round(float(self.planet["StarTeff"].value)))} {self.planet["StarTeff"].unit}'],
                    [ 'lg',  f'{round(float(self.planet["PlanetLogg"].value),1)} {self.planet["PlanetLogg"].unit}', f'{round(float(self.planet["StarLogg"].value),1)} {self.planet["StarLogg"].unit}'],
                    [ 'M',  f'{round(float(self.planet["PlanetMass"].value),1)} {self.planet["PlanetMass"].unit}', f'{round(float(self.planet["StarMass"].value),1)} {self.planet["StarMass"].unit}'],
                    [ 'R',  f'{round(float(self.planet["PlanetRadius"].value),1)} {self.planet["PlanetRadius"].unit}', f'{round(float(self.planet["StarRad"].value),1)} {self.planet["StarRad"].unit}'],
                    [ f'mag',  f'{round(self.mag_planet,1)}', f'{round(self.mag_star,1)}'],
                    [ 'Vsini',  f'{round(float(self.planet["PlanetVsini"].value),1)} {self.planet["PlanetVsini"].unit}', f'{round(float(self.planet["StarVsini"].value),1)} {self.planet["StarVsini"].unit}']]
        self.column_headers = self.data.pop(0)
        self.row_headers = [x.pop(0) for x in self.data]
        self.cell_text = []
        for row in self.data:
            self.cell_text.append([x for x in row])
        self.rcolors = plt.cm.Oranges(np.full(len(self.row_headers), 0.333))
        self.ccolors = plt.cm.Oranges(np.full(len(self.column_headers), 0.333))
        self.table_parameters = self.__plt2.table(cellText=self.cell_text,rowLabels=self.row_headers,rowColours=self.rcolors,rowLoc='center',colColours=self.ccolors,colLabels=self.column_headers,loc='center',bbox = [0.12, 0., 0.89, 0.4])
        self.table_parameters.set_fontsize(16) 
        self.cells = self.table_parameters.properties()["celld"] ; self.col=len(self.column_headers) ; self.rows=len(self.row_headers)
        for i in range(self.col):
            for j in range(self.rows+1):
                self.cells[j, i].set_text_props(ha="center")
        self.data =  [[ 'S/N', f'{round(self.SNR[self.planet_index],1)}'],
                    [ 'Flux ratio', '{0:.1e}'.format(self.flux_ratio)],
                    [ 'Angular separation', f"{round(float(self.planet['AngSep'].value))} {self.planet['AngSep'].unit}"],
                    [ 'SMA', f"{round(float(self.planet['SMA'].value),1)} {self.planet['SMA'].unit}"],
                    [ f'Discovery method', self.planet["DiscoveryMethod"]],
                    [ f'Star spectral type', f"{self.planet['StarSpT']}"],
                    [ f'Delta radial velocity', f"{round(float(self.planet['DeltaRadialVelocity'].value),1)} {self.planet['DeltaRadialVelocity'].unit}"],
                    [ f'Star radial velocity', f"{round(float(self.planet['StarRadialVelocity'].value),1)} {self.planet['StarRadialVelocity'].unit}"],
                    [ f'Inclination', f"{round(float(self.planet['Inc'].value))} °"],
                    [ f'Distance', f"{round(float(self.planet['Distance'].value),1)} {self.planet['Distance'].unit}"]]
        self.row_headers = [x.pop(0) for x in self.data]
        self.cell_text = []
        for row in self.data:
            self.cell_text.append([x for x in row])
        self.rcolors = plt.cm.Oranges(np.full(len(self.row_headers), 0.333))
        self.table_parameters = self.__plt2.table(cellText=self.cell_text,rowLabels=self.row_headers,rowColours=self.rcolors,rowLoc='center',loc='center',bbox = [0.49, 0.47, 0.52, 0.45])
        self.table_parameters.set_fontsize(16) 
        self.cells = self.table_parameters.properties()["celld"] ; self.col=len(self.column_headers) ; self.rows=len(self.row_headers)
        for i in range(-1,1):
            for j in range(self.rows):
                self.cells[j, i].set_text_props(ha="center")
        self.__plt2.text(1.05, -0.05, self.date, horizontalalignment='right', size=14, weight='light')
        
    def open_popup(self):
       self.popup_state = "Open"
       self.popup = Toplevel(self)
       self.popup.title(self.planet["PlanetName"]) ; self.popup.attributes("-topmost", True)
       Button(self.popup, text="S/N", command=lambda *args:self.SNR_calculation(),bg="dark orange",fg="black",font=('Arial', 12, 'bold')).grid(row=11, column=0, sticky="nsew")
       Button(self.popup, text="Contrast", command=lambda *args:self.contrast_calculation(),bg="dark orange",fg="black",font=('Arial', 12, 'bold')).grid(row=11, column=2, sticky="nsew")
       self.popup.protocol('WM_DELETE_WINDOW',lambda: self.onclose())
    def onclose(self): # Func to be called when window is closing, passing the window name
        self.popup_state = "Close" # Set it to close
        self.popup.destroy() # Destroy the window

    def SNR_calculation(self):
        self.calculation = "SNR"
        self.FastCurves_calculation()
    def contrast_calculation(self):
        self.calculation = "contrast"
        self.FastCurves_calculation()
    def FastCurves_calculation(self):
        planet_spectrum , planet_thermal , planet_reflected , star_spectrum = thermal_reflected_spectrum(self.planet,self.instru,thermal_model=self.thermal_model,reflected_model=self.reflected_model,wave=None,vega_spectrum=None,show=True)
        if self.spectrum_contributions == "thermal": 
            planet_spectrum = planet_thermal
        elif self.spectrum_contributions == "reflected":
            planet_spectrum = planet_reflected
        mag_p = self.planet["PlanetINSTRUmag("+self.instru+")("+self.spectrum_contributions+")"]
        mag_s = self.planet["StarINSTRUmag("+self.instru+")"]
        band0= "instru"
        #FastCurves(instru=self.instru,calculation=self.calculation,T_planet=float(self.planet["PlanetTeq"].value),lg_planet=float(self.planet["PlanetLogg"].value),mag_star=mag_s,band0=band0,T_star=float(self.planet["StarTeff"].value),lg_star=float(self.planet["StarLogg"].value),star_rv=float(self.planet["StarRadialVelocity"].value),delta_rv=float(self.planet["DeltaRadialVelocity"].value),vsini_planet=float(self.planet["PlanetVsini"].value),vsini_star=float(self.planet["StarVsini"].value),exposure_time=self.exposure_time.get(),model=self.model,mag_planet=mag_p,separation_planet=float(self.planet["AngSep"].value/1000),name_planet=self.planet["PlanetName"],systematic=self.systematic,show_plot=True,print_value=True,star_spectrum=star_spectrum,planet_spectrum=planet_spectrum,apodizer=self.apodizer,strehl=self.strehl)
        FastCurves(instru=self.instru,calculation=self.calculation,T_planet=float(self.planet["PlanetTeq"].value),lg_planet=float(self.planet["PlanetLogg"].value),mag_star=mag_s,band0=band0,T_star=float(self.planet["StarTeff"].value),lg_star=float(self.planet["StarLogg"].value),exposure_time=self.exposure_time.get(),model=self.model,mag_planet=mag_p,separation_planet=float(self.planet["AngSep"].value/1000),name_planet=self.planet["PlanetName"],systematic=self.systematic,show_plot=True,print_value=True,star_spectrum=star_spectrum,planet_spectrum=planet_spectrum,apodizer=self.apodizer,strehl=self.strehl)

    def destroy_lower_buttons(self):
        try :
            self.button_band_calculation.destroy()
        except :
            pass
        try :
            self.button_model.destroy()
        except :
            pass
        try :
            self.button_systematics.destroy()
        except :
            pass
        try :
            self.button_visible_targets.destroy()
        except :
            pass

def FastYield_interface():
    app = MyWindow()
    app.mainloop()