import sys
import numpy as np
from math import *
import tkinter as tk
from matplotlib.cm import ScalarMappable

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.insert(0, '../src')
from src.FastCurves import *
from src.FastCurves_interface import *
from src.colormaps import *
from src.FastYield import *


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
        self.systematics = False
        self.t_instru = None
        self.popup = None
        self.popup_state = "Close"
        self.calculation = None
        self.planet = None
        
        self.title("FastYield")
        try:
            self.state('zoomed') #works fine on Windows!
        except:
            m = self.maxsize()
            self.geometry('{}x{}+0+0'.format(*m))    
        self.configure(bg='black')
        
        Label(self, text="FastYield", font='Magneto 30 bold',bg="black",fg="dark orange").pack()
        
        # CHOIX DE LA TABLE
        button_table = tk.Frame(self) ; button_table.pack(side=tk.TOP, fill=tk.X)
        self.__btn_archive = tk.Button(button_table, text="ARCHIVE TABLE", command=self.btn_archive_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_archive.grid(row=0, column=0, sticky="nsew") ; button_table.grid_columnconfigure(0, weight=1)
        self.__btn_simulated = tk.Button(button_table, text="SIMULATED TABLE", command=self.btn_simulated_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_simulated.grid(row=0, column=1, sticky="nsew") ; button_table.grid_columnconfigure(1, weight=1)

        # CHOIX DE L'INSTRUMENT 
        button_instru = tk.Frame(self) ; button_instru.pack(side=tk.TOP, fill=tk.X)
        self.__btn_harmoni = tk.Button(button_instru, text="ELT/HARMONI", command=self.btn_harmoni_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_harmoni.grid(row=0, column=0, sticky="nsew") ; button_instru.grid_columnconfigure(0, weight=1)
        self.__btn_eris = tk.Button(button_instru, text="VLT/ERIS", command=self.btn_eris_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_eris.grid(row=0, column=1, sticky="nsew") ; button_instru.grid_columnconfigure(1, weight=1)
        self.__btn_mirimrs = tk.Button(button_instru, text="JWST/MIRI/MRS", command=self.btn_mirimrs_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_mirimrs.grid(row=0, column=2, sticky="nsew") ; button_instru.grid_columnconfigure(2, weight=1)
        self.__btn_nircam = tk.Button(button_instru, text="JWST/NIRSpec", command=self.btn_nirspec_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_nircam.grid(row=0, column=3, sticky="nsew") ; button_instru.grid_columnconfigure(3, weight=1)
        self.__btn_nircam = tk.Button(button_instru, text="JWST/NIRCam", command=self.btn_nircam_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_nircam.grid(row=0, column=4, sticky="nsew") ; button_instru.grid_columnconfigure(4, weight=1)
        
        # CHOIX DES UNITES + texp 
        button_units_texp = tk.Frame(self) ; button_units_texp.pack(side=tk.TOP, fill=tk.X)
        self.__btn_physic = tk.Button(button_units_texp, text="Physical units", command=self.btn_physic_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_physic.grid(row=0, column=0, sticky="nsew") ; button_units_texp.grid_columnconfigure(0, weight=1)
        button_texp = tk.LabelFrame(button_units_texp) ; button_texp.grid(row=0, column=1, sticky="nsew") ; button_texp.grid_columnconfigure(1, weight=1)
        self.__btn_exp = Label(button_texp, text = "Exposure time (in mn) : ",fg="black",font=('Arial', 12, 'bold')) ; self.__btn_exp.grid(column=0, row=0)
        self.__btn_exp = Button(button_texp, text = "Enter",width=10,height=1, command=self.archive_table_plot,bg="dark orange",fg="black",font=('Arial', 12, 'bold')) ; self.__btn_exp.grid(column=2, row=0) 
        self.__btn_entry = Entry(button_texp, width=10,textvariable=self.exposure_time,justify=CENTER,font=('Arial', 12, 'bold')) ; self.__btn_entry.grid(column=1, row=0)
        self.__btn_entry.bind("<Return>", lambda _ : self.archive_table_plot())
        self.__btn_contrast = tk.Button(button_units_texp, text="Observational units", command=self.btn_contrast_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn_contrast.grid(row=0, column=3, sticky="nsew") ; button_units_texp.grid_columnconfigure(3, weight=1)
        
        # On instancie le Canvas MPL.
        self.__fig = Figure() ; self.__canvas = FigureCanvasTkAgg(self.__fig, master=self) ; self.__canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1) ; self.__canvas.mpl_connect("button_press_event", self.canvas_clicked)
        self.__plt = self.__fig.add_subplot(111) ; self.__plt.tick_params(axis='both', labelsize=16)
        cmap = plt.get_cmap("rainbow") ; norm = plt.Normalize(0,5) ; sm =  ScalarMappable(norm=norm, cmap=cmap) ; sm.set_array([])
        self.ax = self.__fig.get_axes() ; self.__cbar = self.__fig.colorbar(sm,ax=self.ax) ; self.__cbar.set_label('SNR', fontsize=20, labelpad=20, rotation=270) ; self.__cbar.ax.tick_params(labelsize=16)
        
        # On choisit l'instrument par défaut
        self.btn_harmoni_clicked()
    
    def get_latitude(self):
        if self.instru == "HARMONI" or self.instru=="ANDES":
            self.latitude = -24.589317 # °N
        elif self.instru == "ERIS":
            self.latitude = -24.627167 # °N
            
    def btn_archive_clicked(self):
        self.destroy_lower_buttons()
        self.table = "Archive"
        if self.instru == "HARMONI":
            self.btn_harmoni_clicked()
        elif self.instru == "ERIS":
            self.btn_eris_clicked()
        elif self.instru == "MIRIMRS":
            self.btn_mirimrs_clicked()
        elif self.instru == "NIRCam":
            self.btn_nircam_clicked()
    def btn_simulated_clicked(self):
        self.destroy_lower_buttons()
        self.table = "Simulated"
        if self.instru == "HARMONI":
            self.btn_harmoni_clicked()
        elif self.instru == "ERIS":
            self.btn_eris_clicked()
        elif self.instru == "MIRIMRS":
            self.btn_mirimrs_clicked()
        elif self.instru == "NIRCam":
            self.btn_nircam_clicked()

    def btn_harmoni_clicked(self):
        self.destroy_lower_buttons()
        self.instru = "HARMONI"
        self.config_data = get_config_data(self.instru)
        self.apodizer = "NO_SP"
        self.strehl = "JQ1"
        self.systematics = False
        self.thermal_model = "BT-Settl"
        self.reflected_model = "PICASO"
        self.model = self.thermal_model+"+"+self.reflected_model
        self.spectrum_contributions = "thermal+reflected"
        self.band = "INSTRU"
        self.create_button_band()
        if self.table == "Archive":
            self.create_button_model()
            self.create_button_visisble_targets()
        self.archive_table_plot()
    def btn_eris_clicked(self):
        self.destroy_lower_buttons()
        self.instru = "ERIS"
        self.config_data = get_config_data(self.instru)
        self.apodizer = "NO_SP"
        self.strehl = "JQ0"
        self.systematics = False
        self.thermal_model = "BT-Settl"
        self.reflected_model = "PICASO"
        self.model = self.thermal_model+"+"+self.reflected_model
        self.spectrum_contributions = "thermal+reflected"
        self.band = "INSTRU"
        self.create_button_band()
        if self.table == "Archive":
            self.create_button_model()
            self.create_button_visisble_targets()
        self.archive_table_plot()
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
        self.create_button_band()
        if self.table == "Archive":
            self.create_button_model()
        self.button_systematics = tk.Frame(self) ; self.button_systematics.pack(side=tk.TOP, fill=tk.X)
        self.__btn = tk.Button(self.button_systematics, text="Without systematics", command=self.btn_no_systematics_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=0, sticky="nsew") ; self.button_systematics.grid_columnconfigure(0, weight=1)
        self.__btn = tk.Button(self.button_systematics, text="With systematics", command=self.btn_systematic_limit_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=1, sticky="nsew") ; self.button_systematics.grid_columnconfigure(1, weight=1)
        self.only_visible_targets = False
        self.archive_table_plot()
    def btn_nirspec_clicked(self):
        self.destroy_lower_buttons()
        self.instru = "NIRSpec"
        self.config_data = get_config_data(self.instru)
        self.apodizer = "NO_SP"
        self.strehl = "NO_JQ"
        self.systematics = False
        self.thermal_model = "BT-Settl"
        self.reflected_model = "PICASO"
        self.model = self.thermal_model+"+"+self.reflected_model
        self.spectrum_contributions = "thermal+reflected"
        self.band = "INSTRU"
        self.create_button_band()
        if self.table == "Archive":
            self.create_button_model()
        self.only_visible_targets = False
        self.archive_table_plot()
    def btn_nircam_clicked(self):
        self.destroy_lower_buttons()
        self.instru = "NIRCam"
        self.config_data = get_config_data(self.instru)
        self.apodizer = "NO_SP"
        self.strehl = "NO_JQ"
        self.systematics = False
        self.thermal_model = "BT-Settl"
        self.reflected_model = "PICASO"
        self.model = self.thermal_model+"+"+self.reflected_model
        self.spectrum_contributions = "thermal+reflected"
        self.band = "INSTRU"
        self.create_button_band()
        if self.table == "Archive":
            self.create_button_model()
        self.only_visible_targets = False
        self.archive_table_plot()
        
    def btn_no_systematics_clicked(self):
        self.systematics = False
        self.archive_table_plot()
    def btn_systematic_limit_clicked(self):
        self.systematics = True
        self.archive_table_plot()
    
    def create_button_band(self):
        list_band = ["INSTRU"]
        for name_band in self.config_data['gratings'] :
            list_band.append(name_band)
        self.button_band = tk.Frame(self) ; self.button_band.pack(side=tk.TOP, fill=tk.X)
        self.button_bands = tk.LabelFrame(self.button_band,bg="dark orange") ; self.button_bands.grid(row=0, column=0, sticky="nsew") ; self.button_band.grid_columnconfigure(0, weight=1)
        self.__txt_bands = Label(self.button_bands, text = "Bandwidth : ",fg="black",bg="dark orange",font=('Arial', 14, 'bold')) ; self.__txt_bands.grid(column=0, row=0) ; self.button_bands.grid_columnconfigure(0, weight=1)
        self.__list_bands = ttk.Combobox(self.button_bands,state='readonly',font=('Arial', 14,'bold'),justify='center') ; self.__list_bands.grid(column=1, row=0) ; self.button_bands.grid_columnconfigure(1, weight=1)
        popdown = self.__list_bands.tk.eval('ttk::combobox::PopdownWindow %s' % self.__list_bands)
        self.__list_bands.tk.call('%s.f.l' % popdown, 'configure', '-font', self.__list_bands['font'])
        self.__list_bands['values'] = list_band
        self.__list_bands.current(0) #index de l'élément sélectionné
        self.__list_bands.bind("<<ComboboxSelected>>", lambda _ : self.band_clicked())
    
    def band_clicked(self):
        self.band = self.__list_bands.get()
        self.archive_table_plot()

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
            #if self.config_data["lambda_range"]["lambda_max"] < 3.5 :
            self.__list_reflected_model['values'] = ("None","PICASO","tellurics","flat")
            #else :
                #self.__list_reflected_model['values'] = ("None","PICASO","flat")
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
        self.archive_table_plot()
        
    def create_button_visisble_targets(self):
        # CHOIX DES UNITES + texp
        self.button_visible_targets = tk.Frame(self) ; self.button_visible_targets.pack(side=tk.TOP, fill=tk.X)
        self.__btn = tk.Button(self.button_visible_targets, text="All targets", command=self.btn_all_targets_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=0, sticky="nsew") ; self.button_visible_targets.grid_columnconfigure(0, weight=1)
        self.__btn = tk.Button(self.button_visible_targets, text="Only visible targets from the observation site", command=self.btn_visible_targets_clicked,bg="dark orange",fg="black", font=('Arial', 12, 'bold')) ; self.__btn.grid(row=0, column=1, sticky="nsew") ; self.button_visible_targets.grid_columnconfigure(1, weight=1)
        self.button_hmin = tk.LabelFrame(self.button_visible_targets) ; self.button_hmin.grid(row=0, column=2, sticky="nsew") ; self.button_hmin.grid_columnconfigure(2, weight=1)
        self.__txt_hmin = Label(self.button_hmin, text = "Minimum elevation (in °) : ",fg="black",font=('Arial', 12, 'bold')) ; self.__txt_hmin.grid(column=0, row=0)
        self.__btn_hmin = Button(self.button_hmin, text = "Enter",width=10,height=1, command=self.archive_table_plot,bg="dark orange",fg="black",font=('Arial', 12, 'bold')) ; self.__btn_hmin.grid(column=2, row=0) 
        self.__btn_entry_hmin = Entry(self.button_hmin, width=10,textvariable=self.hmin,justify=CENTER,font=('Arial', 12, 'bold')) ; self.__btn_entry_hmin.grid(column=1, row=0) 
        self.__btn_entry_hmin.bind("<Return>", lambda _ : self.archive_table_plot())
    def btn_all_targets_clicked(self):
        self.only_visible_targets = False
        self.archive_table_plot()
    def btn_visible_targets_clicked(self):
        self.only_visible_targets = True
        self.archive_table_plot()

    def btn_physic_clicked(self):
        self.contrast = False
        self.archive_table_plot()
    def btn_contrast_clicked(self):
        self.contrast = True
        self.archive_table_plot()
        
    def archive_table_plot(self):
        self.__plt.clear()
        try :
            t_exp = self.exposure_time.get()
        except :
            t_exp = 120
        # chargement des tables de planètes
        t_raw = load_planet_table(self.table+"_Pull_for_FastCurves.ecsv")
        if self.table == "Archive":
            t_raw = t_raw[np.logical_not(t_raw['AngSep'].mask)]
        if self.systematics :
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
        z_instru = np.copy(self.SNR) ; z_instru[z_instru>=5]=5 ; nb_detected = len(self.SNR[self.SNR>=5])
        # WORKING ANGLE
        if self.instru == "HARMONI":
            iwa = self.config_data["apodizers"][self.apodizer].sep
        else :
            lambda_c = self.config_data["lambda_range"]["lambda_min"] *1e-6*u.m
            diameter = self.config_data['telescope']['diameter'] *u.m
            iwa = lambda_c/diameter*u.rad ; iwa = iwa.to(u.mas) ; iwa = iwa.value
        owa = self.config_data["spec"]["FOV"]*1000 # en mas
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
            x_raw = t_raw['SMA'] ; self.x_instru  = self.t_instru['SMA']
            y_raw = t_raw["PlanetMass"] ; self.y_instru  = self.t_instru["PlanetMass"]
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
        x_raw_rv_nep = np.array(x_raw[mask_raw_rv_nep]) ; y_raw_rv_nep = np.array(y_raw[mask_raw_rv_nep])
        x_raw_im_jup = np.array(x_raw[mask_raw_im_jup]) ; y_raw_im_jup = np.array(y_raw[mask_raw_im_jup])
        x_raw_tr_ter = np.array(x_raw[mask_raw_tr_ter]) ; y_raw_tr_ter = np.array(y_raw[mask_raw_tr_ter])
        x_raw_ot_sol = np.array(x_raw[mask_raw_ot_sol]) ; y_raw_ot_sol = np.array(y_raw[mask_raw_ot_sol])
        x_instru_rv_nep = self.x_instru[mask_instru_rv_nep] ; y_instru_rv_nep = self.y_instru[mask_instru_rv_nep] ; z_instru_rv_nep = z_instru[mask_instru_rv_nep]
        x_instru_im_jup = self.x_instru[mask_instru_im_jup] ; y_instru_im_jup = self.y_instru[mask_instru_im_jup] ; z_instru_im_jup = z_instru[mask_instru_im_jup]
        x_instru_tr_ter = self.x_instru[mask_instru_tr_ter] ; y_instru_tr_ter = self.y_instru[mask_instru_tr_ter] ; z_instru_tr_ter = z_instru[mask_instru_tr_ter]
        x_instru_ot_sol = self.x_instru[mask_instru_ot_sol] ; y_instru_ot_sol = self.y_instru[mask_instru_ot_sol] ; z_instru_ot_sol = z_instru[mask_instru_ot_sol]
        # PLOT 
        self.__plt.plot([],[],'kv',ms=10,label=label_tr_ter) ; self.__plt.plot([],[],'ko',ms=10,label=label_rv_nep) ; self.__plt.plot([],[],'ks',ms=10,label=label_im_jup) ; self.__plt.plot([],[],'kP',ms=10,label=label_ot_sol)
        self.__plt.plot(x_raw_tr_ter,y_raw_tr_ter,'kv',alpha=0.5,ms=10) ; self.__plt.plot(x_raw_rv_nep,y_raw_rv_nep,'ko',alpha=0.5,ms=10) ; self.__plt.plot(x_raw_im_jup,y_raw_im_jup,'ks',alpha=0.5,ms=10) ; self.__plt.plot(x_raw_ot_sol,y_raw_ot_sol,'kP',alpha=0.5,ms=10)
        self.__plt.scatter(x_instru_tr_ter, y_instru_tr_ter, s=100+100*z_instru_tr_ter/5, c=z_instru_tr_ter, ec="k",marker="v",cmap="rainbow",vmin=0,vmax=5,zorder=3)
        self.__plt.scatter(x_instru_rv_nep, y_instru_rv_nep,s=100+100*z_instru_rv_nep/5, c=z_instru_rv_nep, ec="k", marker="o",cmap="rainbow",vmin=0,vmax=5,zorder=3)
        self.__plt.scatter(x_instru_im_jup, y_instru_im_jup,s=100+100*z_instru_im_jup/5, c=z_instru_im_jup, ec="k",marker="s",cmap="rainbow",vmin=0,vmax=5,zorder=3)
        self.__plt.scatter(x_instru_ot_sol, y_instru_ot_sol,s=100+100*z_instru_ot_sol/5, c=z_instru_ot_sol, ec="k",marker="P",cmap="rainbow",vmin=0,vmax=5,zorder=3)
        if self.band == "INSTRU":
            self.lmin = self.config_data["lambda_range"]["lambda_min"] ; self.lmax = self.config_data["lambda_range"]["lambda_max"]
        else :
            self.lmin = self.config_data['gratings'][self.band].lmin ; self.lmax = self.config_data['gratings'][self.band].lmax
        if self.systematics:
            txt_syst = "(with systematics)"
        else :
            txt_syst = "(without systematics)"
        if self.band=="INSTRU":
            self.__plt.set_title(f'Known exoplanets detection yield with {self.instru} ({self.spectrum_contributions} with {self.model})'+'\n on '+self.band+'-band (from '+str(round(self.lmin,1))+' to '+str(round(self.lmax,1))+f' µm) with '+'$t_{exp}$=' + str(round(t_exp)) + 'mn ' + txt_syst, fontsize = 24)
        else :
            self.__plt.set_title(f'Known exoplanets detection yield with {self.instru} ({self.spectrum_contributions} with {self.model})'+'\n on '+self.band+'-band (from '+str(round(self.lmin,1))+' to '+str(round(self.lmax,1))+f' µm with R ~ {round(self.config_data["gratings"][self.band].R)}) with '+'$t_{exp}$=' + str(round(t_exp)) + 'mn ' + txt_syst, fontsize = 24)
        if self.contrast :
            self.__plt.set_xlabel(f'Angular separation (in {x_raw.unit})', fontsize = 20)
            self.__plt.set_ylabel(f'Contrast (on {self.band}-band)', fontsize = 20)
            self.__plt.axvspan(iwa, owa, color='k', alpha=0.5, lw=0,label="Working angle",zorder=2)
        else :
            self.__plt.set_xlabel(f'Semi Major Axis (in {x_raw.unit})', fontsize = 20)
            self.__plt.set_ylabel(f"Planet mass (in {y_raw.unit})", fontsize = 20)
            self.__plt.set_ylim(1e-1,2.5e4) ; self.__plt.set_xlim(5e-3,1e4)
        self.__plt.legend(loc="lower right",fontsize=16)
        self.__plt.text(.01, .99, f'total number of planets = {len(t_raw)+len(self.t_instru)}', ha='left', va='top', transform=self.__plt.transAxes,fontsize=12)
        self.__plt.text(.01, .965, f'number of planets detected = {nb_detected}', ha='left', va='top', transform=self.__plt.transAxes,fontsize=12)
        if self.planet is not None :
            if self.planet["PlanetName"] in self.t_instru["PlanetName"] : 
                self.planet_index = planet_index(self.t_instru,self.planet["PlanetName"])
                self.planet = self.t_instru[self.planet_index]
                if self.popup_state == "Open" :
                    self.popup.destroy()
                    self.open_popup()
                self.__plt.plot(self.x_instru[self.planet_index], self.y_instru[self.planet_index],"kX",ms=14,zorder=4)
            else :
                if self.popup is not None :
                    self.popup_state = "Close"
                    self.popup.destroy()
        self.__plt.set_yscale('log') ; self.__plt.set_xscale('log') ; self.__plt.grid(True) ; self.__canvas.draw()
        
    def canvas_clicked(self,event):
        if event.xdata is not None and event.ydata is not None :
            if self.popup is not None :
                self.popup.destroy()
            c_x = floor(np.log10(event.xdata)) # https://neutrium.net/general-engineering/accurate-readings-from-log-plots/
            c_y = floor(np.log10(event.ydata))
            a_clicked_x = (np.log10(event.xdata/10**(c_x)))
            a_clicked_y = (np.log10(event.ydata/10**(c_y)))
            a_data_x = (np.log10(np.array(self.x_instru.value)/10**(c_x)))
            a_data_y = (np.log10(np.array(self.y_instru.value)/10**(c_y)))
            delta_x = np.abs(a_clicked_x - a_data_x)
            delta_y = np.abs(a_clicked_y - a_data_y)
            self.planet_index = ((delta_x)**2+(delta_y)**2).argmin()
            self.planet = self.t_instru[self.planet_index]
            self.open_popup()
            self.archive_table_plot()
        
    def open_popup(self):
       self.popup_state = "Open"
       self.popup = Toplevel(self)
       self.popup.title(self.planet["PlanetName"]) ; self.popup.attributes("-topmost", True)
       
       Label(self.popup, text="SNR ("+self.band+f") = {round(self.SNR[self.planet_index],1)}",font=('Arial', 12, 'bold')).grid(column=0, row=0, sticky="nsew")
       Label(self.popup, text="|",font=('Arial', 12, 'bold')).grid(column=1, row=0, sticky="nsew")
       if self.band != "INSTRU":
           Label(self.popup, text="flux ratio ("+self.band+") = {0:.2e}".format(10**(-(self.planet["Planet"+self.band+"mag("+self.spectrum_contributions+")"]-np.array(self.planet["Star"+self.band+"mag"]))/2.5)*u.dimensionless_unscaled)+" ("+self.spectrum_contributions+")",font=('Arial', 12, 'bold')).grid(column=2, row=0, sticky="nsew")
       else :
           Label(self.popup, text="flux ratio ("+self.band+") = {0:.2e}".format(10**(-(self.planet["Planet"+self.band+"mag("+self.instru+")("+self.spectrum_contributions+")"]-np.array(self.planet["Star"+self.band+"mag("+self.instru+")"]))/2.5)*u.dimensionless_unscaled)+" ("+self.spectrum_contributions+")",font=('Arial', 12, 'bold')).grid(column=2, row=0, sticky="nsew")

       Label(self.popup, text=f"AngSep = {round(float(self.planet['AngSep'].value))} {self.planet['AngSep'].unit}",font=('Arial', 12, 'bold')).grid(column=0, row=1, sticky="nsew")
       Label(self.popup, text="|",font=('Arial', 12, 'bold')).grid(column=1, row=1, sticky="nsew")
       Label(self.popup, text=f"SMA = {round(float(self.planet['SMA'].value),1)} {self.planet['SMA'].unit}",font=('Arial', 12, 'bold')).grid(column=2, row=1, sticky="nsew")
       
       Label(self.popup, text=f"T(planet) = {round(float(self.planet['PlanetTeq'].value))} {self.planet['PlanetTeq'].unit}",font=('Arial', 12, 'bold')).grid(column=0, row=2, sticky="nsew")
       Label(self.popup, text="|",font=('Arial', 12, 'bold')).grid(column=1, row=2, sticky="nsew")
       Label(self.popup, text=f"T(star) = {round(float(self.planet['StarTeff'].value))} {self.planet['StarTeff'].unit}",font=('Arial', 12, 'bold')).grid(column=2, row=2, sticky="nsew")
       
       Label(self.popup, text=f"lg(planet) = {round(float(self.planet['PlanetLogg'].value),2)}",font=('Arial', 12, 'bold')).grid(column=0, row=3, sticky="nsew")
       Label(self.popup, text="|",font=('Arial', 12, 'bold')).grid(column=1, row=3, sticky="nsew")
       Label(self.popup, text=f"lg(star) = {round(float(self.planet['StarLogg'].value),2)}",font=('Arial', 12, 'bold')).grid(column=2, row=3, sticky="nsew")
       
       Label(self.popup, text=f"M(planet) = {round(float(self.planet['PlanetMass'].value))} {self.planet['PlanetMass'].unit}",font=('Arial', 12, 'bold')).grid(column=0, row=4, sticky="nsew")
       Label(self.popup, text="|",font=('Arial', 12, 'bold')).grid(column=1, row=4, sticky="nsew")
       Label(self.popup, text=f"M(star) = {round(float(self.planet['StarMass'].value),1)} {self.planet['StarMass'].unit}",font=('Arial', 12, 'bold')).grid(column=2, row=4, sticky="nsew")
       
       Label(self.popup, text=f"R(planet) = {round(float(self.planet['PlanetRadius'].value),1)} {self.planet['PlanetRadius'].unit}",font=('Arial', 12, 'bold')).grid(column=0, row=5, sticky="nsew")
       Label(self.popup, text="|",font=('Arial', 12, 'bold')).grid(column=1, row=5, sticky="nsew")
       Label(self.popup, text=f"R(star) = {round(float(self.planet['StarRad'].value),1)} {self.planet['StarRad'].unit}",font=('Arial', 12, 'bold')).grid(column=2, row=5, sticky="nsew")
       
       if self.band != "INSTRU":
           Label(self.popup, text=f"mag(planet,"+self.band+f") = {round(float(self.planet['Planet'+self.band+'mag('+self.spectrum_contributions+')']),3)} ("+self.spectrum_contributions+")",font=('Arial', 12, 'bold')).grid(column=0, row=6, sticky="nsew")
           Label(self.popup, text="|",font=('Arial', 12, 'bold')).grid(column=1, row=6, sticky="nsew")
           Label(self.popup, text=f"mag(star,"+self.band+f") = {round(float(self.planet['Star'+self.band+'mag']),3)} ",font=('Arial', 12, 'bold')).grid(column=2, row=6, sticky="nsew")
       else :
           Label(self.popup, text=f"mag(planet,"+self.band+f") = {round(float(self.planet['Planet'+self.band+'mag('+self.instru+')('+self.spectrum_contributions+')']),3)} ("+self.spectrum_contributions+")",font=('Arial', 12, 'bold')).grid(column=0, row=6, sticky="nsew")
           Label(self.popup, text="|",font=('Arial', 12, 'bold')).grid(column=1, row=6, sticky="nsew")
           Label(self.popup, text=f"mag(star,"+self.band+f") = {round(float(self.planet['Star'+self.band+'mag('+self.instru+')']),3)} ",font=('Arial', 12, 'bold')).grid(column=2, row=6, sticky="nsew")
       
       Label(self.popup, text=f"Discovery Method = {self.planet['DiscoveryMethod']} ",font=('Arial', 12, 'bold')).grid(column=0, row=7, sticky="nsew")
       Label(self.popup, text="|",font=('Arial', 12, 'bold')).grid(column=1, row=7, sticky="nsew")
       Label(self.popup, text=f"Star spectral type = {self.planet['StarSpT']} ",font=('Arial', 12, 'bold')).grid(column=2, row=7, sticky="nsew")
       
       Label(self.popup, text=f"Δrv = {round(float(self.planet['DeltaRadialVelocity'].value),1)} km/s",font=('Arial', 12, 'bold')).grid(column=0, row=8, sticky="nsew")
       Label(self.popup, text="|",font=('Arial', 12, 'bold')).grid(column=1, row=8, sticky="nsew")
       Label(self.popup, text=f"syst_rv = {round(float(self.planet['StarRadialVelocity'].value),1)} km/s",font=('Arial', 12, 'bold')).grid(column=2, row=8, sticky="nsew")
       
       Label(self.popup, text=f"Inclination = {round(float(self.planet['Inc'].value))} °",font=('Arial', 12, 'bold')).grid(column=0, row=9, sticky="nsew")
       Label(self.popup, text="|",font=('Arial', 12, 'bold')).grid(column=1, row=9, sticky="nsew")
       Label(self.popup, text=f"Distance = {round(float(self.planet['Distance'].value),1)} {self.planet['Distance'].unit}",font=('Arial', 12, 'bold')).grid(column=2, row=9, sticky="nsew")
      
       Button(self.popup, text="SNR", command=lambda *args:self.SNR_calculation(),bg="dark orange",fg="black",font=('Arial', 12, 'bold')).grid(row=10, column=0, sticky="nsew")
       Button(self.popup, text="Contrast", command=lambda *args:self.contrast_calculation(),bg="dark orange",fg="black",font=('Arial', 12, 'bold')).grid(row=10, column=2, sticky="nsew")

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
        if self.instru == "HARMONI":
            harmoni(calculation=self.calculation,T_planet=float(self.planet["PlanetTeq"].value),lg_planet=float(self.planet["PlanetLogg"].value),mag_star=mag_s,band0=band0,T_star=float(self.planet["StarTeff"].value),lg_star=float(self.planet["StarLogg"].value),syst_radial_velocity=float(self.planet["StarRadialVelocity"].value),delta_radial_velocity=float(self.planet["DeltaRadialVelocity"].value),star_broadening=float(self.planet["StarVsini"].value),exposure_time=self.exposure_time.get(),model=self.model,mag_planet=mag_p,separation_planet=float(self.planet["AngSep"].value/1000),name_planet=self.planet["PlanetName"],systematic=self.systematics,show_plot=True,print_value=True,star_spectrum=star_spectrum,planet_spectrum=planet_spectrum,apodizer=self.apodizer,strehl=self.strehl)
        elif self.instru=="ERIS":
            eris(calculation=self.calculation,T_planet=float(self.planet["PlanetTeq"].value),lg_planet=float(self.planet["PlanetLogg"].value),mag_star=mag_s,band0=band0,T_star=float(self.planet["StarTeff"].value),lg_star=float(self.planet["StarLogg"].value),syst_radial_velocity=float(self.planet["StarRadialVelocity"].value),delta_radial_velocity=float(self.planet["DeltaRadialVelocity"].value),star_broadening=float(self.planet["StarVsini"].value),exposure_time=self.exposure_time.get(),model=self.model,mag_planet=mag_p,separation_planet=float(self.planet["AngSep"].value/1000),name_planet=self.planet["PlanetName"],systematic=self.systematics,show_plot=True,print_value=True,star_spectrum=star_spectrum,planet_spectrum=planet_spectrum)
        elif self.instru=="MIRIMRS":
            mirimrs(calculation=self.calculation,T_planet=float(self.planet["PlanetTeq"].value),lg_planet=float(self.planet["PlanetLogg"].value),mag_star=mag_s,band0=band0,T_star=float(self.planet["StarTeff"].value),lg_star=float(self.planet["StarLogg"].value),syst_radial_velocity=float(self.planet["StarRadialVelocity"].value),delta_radial_velocity=float(self.planet["DeltaRadialVelocity"].value),star_broadening=float(self.planet["StarVsini"].value),exposure_time=self.exposure_time.get(),model=self.model,mag_planet=mag_p,separation_planet=float(self.planet["AngSep"].value/1000),name_planet=self.planet["PlanetName"],systematic=self.systematics,show_plot=True,print_value=True,star_spectrum=star_spectrum)
        elif self.instru=="NIRCam":
            nircam(calculation=self.calculation,T_planet=float(self.planet["PlanetTeq"].value),lg_planet=float(self.planet["PlanetLogg"].value),mag_star=mag_s,band0=band0,T_star=float(self.planet["StarTeff"].value),lg_star=float(self.planet["StarLogg"].value),syst_radial_velocity=float(self.planet["StarRadialVelocity"].value),delta_radial_velocity=float(self.planet["DeltaRadialVelocity"].value),star_broadening=float(self.planet["StarVsini"].value),exposure_time=self.exposure_time.get(),model=self.model,mag_planet=mag_p,separation_planet=float(self.planet["AngSep"].value/1000),name_planet=self.planet["PlanetName"],systematic=self.systematics,show_plot=True,print_value=True,star_spectrum=star_spectrum,planet_spectrum=planet_spectrum)
        elif self.instru=="NIRSpec":
            nirspec(calculation=self.calculation,T_planet=float(self.planet["PlanetTeq"].value),lg_planet=float(self.planet["PlanetLogg"].value),mag_star=mag_s,band0=band0,T_star=float(self.planet["StarTeff"].value),lg_star=float(self.planet["StarLogg"].value),syst_radial_velocity=float(self.planet["StarRadialVelocity"].value),delta_radial_velocity=float(self.planet["DeltaRadialVelocity"].value),star_broadening=float(self.planet["StarVsini"].value),exposure_time=self.exposure_time.get(),model=self.model,mag_planet=mag_p,separation_planet=float(self.planet["AngSep"].value/1000),name_planet=self.planet["PlanetName"],systematic=self.systematics,show_plot=True,print_value=True,star_spectrum=star_spectrum,planet_spectrum=planet_spectrum)
    
    def destroy_lower_buttons(self):
        try :
            self.button_band.destroy()
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