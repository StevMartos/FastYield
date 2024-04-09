from tkinter import *
from tkinter import ttk
from ttkwidgets.autocomplete import AutocompleteEntry
from src.FastCurves import *
from src.FastYield import *


def FastCurves_interface():
    class under_frame:
    
        def __init__(self):
            self.frame = None
            self.rows = 5
            self.cols = 5
            self.grid_places = [[None]*self.cols]*self.rows
            self.variable = None
    
        def clear_frame(self):
            if self.frame is not None :
                self.frame.destroy()
    
        def instrument_options(self):
            if calculation.get()=="SNR":
                if instrument.get()!="MIRIMRS":
                    frame2_calculation.grid_places[1][1].destroy()
                frame2_calculation.select_SNR()
            self.clear_frame()
            self.frame = LabelFrame(frame1, text="Options", font='Helvetica 10 bold')
            self.frame.grid(column=1, row=0)
            self.grid_places[0][0] = Label(self.frame, text="Coronagraph :", relief=GROOVE,font='Helvetica 8 bold', padx=10, pady=10) ; self.grid_places[0][0].grid(column=0, row=0)
            self.grid_places[1][0] = Label(self.frame, text="Apodizer :", relief=GROOVE,font='Helvetica 8 bold', padx=10, pady=10) ; self.grid_places[1][0].grid(column=1, row=0)
            self.grid_places[2][0] = Label(self.frame, text="Strehl :", relief=GROOVE,font='Helvetica 8 bold', padx=10, pady=10) ; self.grid_places[2][0].grid(column=2, row=0)
            if instrument.get()=="HARMONI":
                coronagraph.set("None") ; apodizer.set("NO_SP") ; strehl.set("JQ1")
                self.grid_places[0][1] = Radiobutton(self.frame,text='None', value='None', variable=coronagraph) ; self.grid_places[0][1].grid(column=0, row=1)
                self.grid_places[1][1] = Radiobutton(self.frame,text='None', value='NO_SP', variable=apodizer) ; self.grid_places[1][1].grid(column=1, row=1)
                self.grid_places[1][2] = Radiobutton(self.frame,text='SP1', value='SP1', variable=apodizer) ; self.grid_places[1][2].grid(column=1, row=2)
                self.grid_places[1][3] = Radiobutton(self.frame,text='SP2', value='SP2', variable=apodizer) ; self.grid_places[1][3].grid(column=1, row=3)
                self.grid_places[2][1] = Radiobutton(self.frame,text='None', value='NO_Strehl', variable=strehl); self.grid_places[2][1].grid(column=2, row=1)
                self.grid_places[2][2] = Radiobutton(self.frame,text='JQ0', value='JQ0', variable=strehl) ; self.grid_places[2][2].grid(column=2, row=2)
                self.grid_places[2][3]= Radiobutton(self.frame,text='JQ1', value='JQ1', variable=strehl) ; self.grid_places[2][3].grid(column=2, row=3)
                self.grid_places[2][4] = Radiobutton(self.frame,text='MED', value='MED', variable=strehl) ; self.grid_places[2][4].grid(column=2, row=4)
            elif instrument.get()=="ERIS":
                coronagraph.set("None") ; apodizer.set("NO_SP") ; strehl.set("JQ0")
                self.grid_places[0][1] = Radiobutton(self.frame,text='None', value='None', variable=coronagraph) ; self.grid_places[0][1].grid(column=0, row=1)
                self.grid_places[1][1] = Radiobutton(self.frame,text='None', value='NO_SP', variable=apodizer) ; self.grid_places[1][1].grid(column=1, row=1)
                self.grid_places[2][1] = Radiobutton(self.frame,text='JQ0', value='JQ0', variable=strehl) ; self.grid_places[2][1].grid(column=2, row=1)
            elif instrument.get()=="MIRIMRS":
                coronagraph.set("None") ; apodizer.set("NO_SP") ; strehl.set("NO_Strehl")
                self.grid_places[0][1] = Radiobutton(self.frame,text='None', value='None', variable=coronagraph) ; self.grid_places[0][1].grid(column=0, row=1)
                self.grid_places[1][1] = Radiobutton(self.frame,text='None', value='NO_SP', variable=apodizer) ; self.grid_places[1][1].grid(column=1, row=1)
                self.grid_places[2][1] = Radiobutton(self.frame,text='None', value='NO_Strehl', variable=strehl) ; self.grid_places[2][1].grid(column=2, row=1)
            elif instrument.get()=="NIRCam":
                coronagraph.set("MASK335R") ; apodizer.set("NO_SP") ; strehl.set("NO_Strehl")
                self.grid_places[0][1] = Radiobutton(self.frame,text='MASK335R', value='MASK335R', variable=coronagraph) ; self.grid_places[0][1].grid(column=0, row=1)
                self.grid_places[1][1] = Radiobutton(self.frame,text='None', value='NO_SP', variable=apodizer) ; self.grid_places[1][1].grid(column=1, row=1)
                self.grid_places[2][1] = Radiobutton(self.frame,text='None', value='NO_Strehl', variable=strehl) ; self.grid_places[2][1].grid(column=2, row=1)
            try :
                frame3_band0.enter_band0()
            except :
                pass
            
        def calculation_choice(self):
            self.clear_frame()
            self.frame = LabelFrame(frame2, text="Contrast or SNR curves", font='Helvetica 10 bold')
            self.frame.grid(column=0, row=0, padx=340)
            self.grid_places[0][0] = Radiobutton(self.frame,text='Contrast', value='contrast', variable=calculation, padx=33.5, command=self.calculation_choice) ; self.grid_places[0][0].grid(column=0, row=0)
            self.grid_places[1][0] = Radiobutton(self.frame,text='SNR', value='SNR', variable=calculation, padx=33.5, command=self.select_SNR) ; self.grid_places[1][0].grid(column=1, row=0)
            self.grid_places[0][1] = Checkbutton(self.frame, text='Plot in Δmag', variable=plot_mag) ; self.grid_places[0][1].grid(column=0, row=1)
            try :
                frame3_mag_planet.clear_frame()
            except :
                pass
            
        def select_SNR(self):
            if self.grid_places[0][1] is not None :
                self.grid_places[0][1].destroy()
            if instrument.get()=="MIRIMRS":
                self.grid_places[1][1] = Checkbutton(self.frame, text='SNR per channel', variable=channel) ; self.grid_places[1][1].grid(column=1, row=1)
            else :
                self.grid_places[1][1] = Label(self.frame, text='',pady=3) ; self.grid_places[1][1].grid(column=1, row=1)        
            frame3_mag_planet.mag_planet_choice() ; frame3_mag_planet.enter_mag_planet()
        
        def exposure_time_choice(self):
            self.clear_frame()
            self.frame = LabelFrame(frame2, text="Exposure time (in minutes)", font='Helvetica 10 bold')
            self.frame.grid(column=1, row=0)
            self.grid_places[1][0] = Button(self.frame, text = "Enter",width=10,height=1, command=self.enter_exposure_time,bg="dark orange",fg="black") ; self.grid_places[1][0].grid(column=1, row=0)
            self.grid_places[0][0] = Entry(self.frame, width=10,textvariable=exposure_time) ; self.grid_places[0][0].grid(column=0, row=0)
            self.grid_places[0][0].bind("<Return>", lambda _ : self.enter_exposure_time())
    
        def enter_exposure_time(self):
            if self.grid_places[0][1] is not None :
                self.grid_places[0][1].destroy()
            try :
                exposure_time.get()
                self.grid_places[0][1] = Label(self.frame, text=f"Entered exposure time : {round(exposure_time.get())} mn", fg="green") ; self.grid_places[0][1].grid(column=0, row=1)
            except:
                self.grid_places[0][1] = Label(self.frame, text="Invalid exposure time", fg="red") ; self.grid_places[0][1].grid(column=0, row=1)
    
        def name_planet_choice(self):
            self.clear_frame()
            self.frame = LabelFrame(frame3, text="Planet name (optional)", font='Helvetica 10 bold')
            self.frame.grid(column=1, row=0,padx=110)
            self.grid_places[1][0] = Button(self.frame, text = "Enter",width=10,height=1, command=self.enter_name_planet,bg="dark orange",fg="black") ; self.grid_places[1][0].grid(column=1, row=0)
            self.grid_places[3][0] = Label(self.frame, text="(If the planet name is recognized, \n it will enter all the other parameters, \n e.g. : Kepler-896 b, bet Pic b, ...)", font='Helvetica 8 italic') ; self.grid_places[3][0].grid(column=3, row=0)
            self.grid_places[0][0] = AutocompleteEntry(self.frame, width=20, font='Helvetica 12',completevalues=list(table_FastCurves["PlanetName"]),textvariable=name_planet) ; self.grid_places[0][0].grid(column=0, row=0)
            self.grid_places[0][0].bind("<Return>", lambda _ : self.enter_name_planet())
    
        def enter_name_planet(self):
            if self.grid_places[0][1] is not None :
                self.grid_places[0][1].destroy()
            if name_planet.get() in table_FastCurves["PlanetName"]:
                self.grid_places[0][1] = Label(self.frame, text="(Recognized) Entered planet name : "+name_planet.get(), fg="green") ; self.grid_places[0][1].grid(column=0, row=1)
                idx = planet_index(planet_table=table_FastCurves,planet_name=name_planet.get())
                T_star.set(round(float(table_FastCurves["StarTeff"][idx].value))) ; lg_star.set(round(float(table_FastCurves["StarLogg"][idx].value),1))
                if band0.get() == "K":
                    mag_star.set(round(float(table_FastCurves["StarKmag"][idx]),3))
                T_planet.set(round(float(table_FastCurves["PlanetTeq"][idx].value))) ; lg_planet.set(round(float(table_FastCurves["PlanetLogg"][idx].value),1))
                if band0.get() != "K":
                    band0.set("K") ; frame3_band0.band0_choice() 
                frame3_T_star.enter_T_star() ; frame3_lg_star.enter_lg_star() ; frame3_mag_star.enter_mag_star()
                frame3_T_planet.enter_T_planet() ; frame3_lg_planet.enter_lg_planet()
                frame3_band0.enter_band0()
                if float(table_FastCurves["AngSep"][idx].value) != 0 :
                    separation_planet.set(round(float(table_FastCurves["AngSep"][idx].value/1000),3)) ; frame3_separation_planet.enter_separation_planet()
                else :
                    separation_planet.set()
                idx = planet_index(planet_table=table_FastCurves,planet_name=name_planet.get())
                if band0.get() == "K":
                    mag_planet.set(round(table_FastCurves["PlanetKmag(thermal)"][idx],3))
                frame3_mag_planet.enter_mag_planet()
            else :
                self.grid_places[0][1] = Label(self.frame, text="(Unrecognized) Entered planet name : "+name_planet.get(), fg="red") ; self.grid_places[0][1].grid(column=0, row=1)
    
        def T_star_choice(self):
            self.clear_frame()
            self.frame = LabelFrame(frame3, text="Star temperature (in Kelvin)", font='Helvetica 10 bold')
            self.frame.grid(column=0, row=1)
            self.grid_places[1][0] = Button(self.frame, text = "Enter",width=10,height=1, command=self.enter_T_star,bg="dark orange",fg="black") ; self.grid_places[1][0].grid(column=1, row=0,padx=6.25)
            self.grid_places[0][0] = Entry(self.frame, width=20,textvariable=T_star) ; self.grid_places[0][0].grid(column=0, row=0,padx=6.25)
            self.grid_places[0][0].bind("<Return>", lambda _ : self.enter_T_star())
    
        def enter_T_star(self):
            if self.grid_places[0][1] is not None :
                self.grid_places[0][1].destroy()
            try :
                T_star.get()
                self.grid_places[0][1] = Label(self.frame, text=f"Entered star temperature : {round(T_star.get())} K", fg="green") ; self.grid_places[0][1].grid(column=0, row=1)
            except:
                self.grid_places[0][1] = Label(self.frame, text="Invalid star temperature", fg="red") ; self.grid_places[0][1].grid(column=0, row=1)
    
        def lg_star_choice(self):
            self.clear_frame()
            self.frame = LabelFrame(frame3, text="Star gravity surface (in log(g))", font='Helvetica 10 bold')
            self.frame.grid(column=1, row=1)
            self.grid_places[1][0] = Button(self.frame, text = "Enter",width=10,height=1, command=self.enter_lg_star,bg="dark orange",fg="black") ; self.grid_places[1][0].grid(column=1, row=0,padx=6)
            self.grid_places[0][0] = Entry(self.frame, width=20,textvariable=lg_star) ; self.grid_places[0][0].grid(column=0, row=0,padx=6)
            self.grid_places[0][0].bind("<Return>", lambda _ : self.enter_lg_star())
    
        def enter_lg_star(self):
            if self.grid_places[0][1] is not None :
                self.grid_places[0][1].destroy()
            try :
                lg_star.get()
                self.grid_places[0][1] = Label(self.frame, text=f"Entered star gravity surface : {round(lg_star.get(),1)}", fg="green") ; self.grid_places[0][1].grid(column=0, row=1)
            except:
                self.grid_places[0][1] = Label(self.frame, text="Invalid star gravity surface", fg="red") ; self.grid_places[0][1].grid(column=0, row=1)
    
        def mag_star_choice(self):
            self.clear_frame()
            self.frame = LabelFrame(frame3, text="Star magnitude ", font='Helvetica 10 bold')
            self.frame.grid(column=2, row=1)
            self.grid_places[1][0] = Button(self.frame, text = "Enter",width=10,height=1, command=self.enter_mag_star,bg="dark orange",fg="black") ; self.grid_places[1][0].grid(column=1, row=0,padx=19)
            self.grid_places[0][0] = Entry(self.frame, width=20,textvariable=mag_star) ; self.grid_places[0][0].grid(column=0, row=0,padx=19)
            self.grid_places[0][0].bind("<Return>", lambda _ : self.enter_mag_star())
    
        def enter_mag_star(self):
            if self.grid_places[0][1] is not None :
                self.grid_places[0][1].destroy()
            try :
                mag_star.get()
                self.grid_places[0][1] = Label(self.frame, text=f"Entered star magnitude : {round(mag_star.get(),3)}", fg="green") ; self.grid_places[0][1].grid(column=0, row=1)
            except:
                self.grid_places[0][1] = Label(self.frame, text="Invalid star magnitude", fg="red") ; self.grid_places[0][1].grid(column=0, row=1)
    
        def T_planet_choice(self):
            self.clear_frame()
            self.frame = LabelFrame(frame3, text="Planet temperature (in Kelvin)", font='Helvetica 10 bold')
            self.frame.grid(column=0, row=2)
            self.grid_places[1][0] = Button(self.frame, text = "Enter",width=10,height=1, command=self.enter_T_planet,bg="dark orange",fg="black") ; self.grid_places[1][0].grid(column=1, row=0)
            self.grid_places[0][0] = Entry(self.frame, width=20,textvariable=T_planet) ; self.grid_places[0][0].grid(column=0, row=0)
            self.grid_places[0][0].bind("<Return>", lambda _ : self.enter_T_planet())
    
        def enter_T_planet(self):
            if self.grid_places[0][1] is not None :
                self.grid_places[0][1].destroy()
            try :
                T_planet.get()
                self.grid_places[0][1] = Label(self.frame, text=f"Entered planet temperature : {round(T_planet.get())} K", fg="green") ; self.grid_places[0][1].grid(column=0, row=1)
            except:
                self.grid_places[0][1] = Label(self.frame, text="Invalid planet temperature", fg="red") ; self.grid_places[0][1].grid(column=0, row=1)
    
        def lg_planet_choice(self):
            self.clear_frame()
            self.frame = LabelFrame(frame3, text="Planet gravity surface (in log(g))", font='Helvetica 10 bold')
            self.frame.grid(column=1, row=2)
            self.grid_places[1][0] = Button(self.frame, text = "Enter",width=10,height=1, command=self.enter_lg_planet,bg="dark orange",fg="black") ; self.grid_places[1][0].grid(column=1, row=0)
            self.grid_places[0][0] = Entry(self.frame, width=20,textvariable=lg_planet) ; self.grid_places[0][0].grid(column=0, row=0)
            self.grid_places[0][0].bind("<Return>", lambda _ : self.enter_lg_planet())
    
        def enter_lg_planet(self):
            if self.grid_places[0][1] is not None :
                self.grid_places[0][1].destroy()
            try :
                lg_planet.get()
                self.grid_places[0][1] = Label(self.frame, text=f"Entered planet gravity surface : {round(lg_planet.get(),1)}", fg="green") ; self.grid_places[0][1].grid(column=0, row=1)
            except:
                self.grid_places[0][1] = Label(self.frame, text="Invalid planet gravity surface", fg="red") ; self.grid_places[0][1].grid(column=0, row=1)
    
        def mag_planet_choice(self):
            if calculation.get() == "SNR":
                self.clear_frame()
                self.frame = LabelFrame(frame3, text="Planet magnitude (needed for SNR curves)", font='Helvetica 10 bold')
                self.frame.grid(column=2, row=2)
                self.grid_places[1][0] = Button(self.frame, text = "Enter",width=10,height=1, command=self.enter_mag_planet,bg="dark orange",fg="black") ; self.grid_places[1][0].grid(column=1, row=0)
                self.grid_places[0][0] = Entry(self.frame, width=20,textvariable=mag_planet) ; self.grid_places[0][0].grid(column=0, row=0)
                self.grid_places[0][0].bind("<Return>", lambda _ : self.enter_mag_planet())
    
        def enter_mag_planet(self):
            if calculation.get() == "SNR":
                if self.grid_places[0][1] is not None :
                    self.grid_places[0][1].destroy()
                try :
                    mag_planet.get()
                    self.grid_places[0][1] = Label(self.frame, text=f"Entered planet magnitude : {round(mag_planet.get(),3)}", fg="green") ; self.grid_places[0][1].grid(column=0, row=1)
                except:
                    self.grid_places[0][1] = Label(self.frame, text="Invalid planet magnitude", fg="red") ; self.grid_places[0][1].grid(column=0, row=1)
    
        def band0_choice(self):
            self.clear_frame()
            self.frame = LabelFrame(frame3, text="Spectral band where mags are defined", font='Helvetica 10 bold')
            self.frame.grid(column=0, row=3)
            self.grid_places[0][0] = ttk.Combobox(self.frame,state='readonly')
            self.grid_places[0][0]['values']= ("J","H","Ks","K","L","L'","instru")
            self.grid_places[0][0].current(3) #index de l'élément sélectionné
            self.grid_places[0][0].pack(padx=65)
            self.grid_places[0][0].bind("<<ComboboxSelected>>", lambda _ : self.enter_band0())
            
        def enter_band0(self):
            band0.set(self.grid_places[0][0].get())
            if band0.get()=="J":
                lambda_c=1.215 ; Dlambda=0.26 # en µm
            elif band0.get()=="H":
                lambda_c=1.654 ; Dlambda=0.29 # en µm
            elif band0.get()=='Ks':
                lambda_c=2.157 ; Dlambda=0.32 # en µm
            elif band0.get()=='K':
                lambda_c=2.179 ; Dlambda=0.41 # en µm
            elif band0.get()=="L": 
                lambda_c=3.547 ; Dlambda=0.57 # en µm
            elif band0.get()=="L'": 
                lambda_c=3.761 ; Dlambda=0.65 # en µm
            elif band0.get()=="instru":
                config_data = get_config_data(instrument.get())
                lambda_c=(config_data["lambda_range"]["lambda_max"]+config_data["lambda_range"]["lambda_min"])/2 ; Dlambda=config_data["lambda_range"]["lambda_max"]-config_data["lambda_range"]["lambda_min"] # en µm
            if self.grid_places[0][1] is not None :
                self.grid_places[0][1].destroy()
            self.grid_places[0][1] = Label(self.frame, text="Entered spectral band : "+band0.get()+ f" (~ {lambda_c} µm)", fg="green") ; self.grid_places[0][1].pack()

            
        def separation_planet_choice(self):
            self.clear_frame()
            self.frame = LabelFrame(frame3, text="Planet separation (in arcsec, optional)", font='Helvetica 10 bold')
            self.frame.grid(column=1, row=3)
            self.grid_places[1][0] = Button(self.frame, text = "Enter",width=10,height=1, command=self.enter_separation_planet,bg="dark orange",fg="black") ; self.grid_places[1][0].grid(column=1, row=0,padx=1.5)
            self.grid_places[0][0] = Entry(self.frame, width=20,textvariable=separation_planet) ; self.grid_places[0][0].grid(column=0, row=0,padx=1.5)
            self.grid_places[0][0].bind("<Return>", lambda _ : self.enter_separation_planet())
    
        def enter_separation_planet(self):
            if self.grid_places[0][1] is not None :
                self.grid_places[0][1].destroy()
            try :
                separation_planet.get()
                self.grid_places[0][1] = Label(self.frame, text=f'Entered planet separation : {round(separation_planet.get(),3)} "', fg="green") ; self.grid_places[0][1].grid(column=0, row=1)
            except:
                self.grid_places[0][1] = Label(self.frame, text=f'Invalid planet separation', fg="red") ; self.grid_places[0][1].grid(column=0, row=1)
                sel
    
    def start(fenetre,start_FC):
        fenetre.destroy()
        start_FC.set(True)
        
    #---------------------------------------------------------------------------------
    
    # Table d'archie de planètes
    
    table_FastCurves = load_planet_table("Archive_Pull_for_FastCurves.ecsv")
    
    
    #---------------------------------------------------------------------------------
    
    # OUVERTURE FENETRE
    fenetre = Tk() ; fenetre.configure(bg='black')
    try:
        fenetre.state('zoomed') #works fine on Windows!
    except:
        m = fenetre.maxsize()
        fenetre.geometry('{}x{}+0+0'.format(*m))   
    fenetre.title("FastCurves") ; fenetre.config(bg='black')
    Label(fenetre, text="FastCurves", font='Magneto 30 bold',bg="black",fg="dark orange").pack()
    
    instrument = StringVar(value="HARMONI") ; coronagraph = StringVar(value="None") ; apodizer = StringVar(value="NO_SP") ; strehl = StringVar(value="JQ1")
    calculation = StringVar(value="contrast") ; plot_mag = BooleanVar(value=False) ; channel = BooleanVar(value=False) ; exposure_time = DoubleVar(value=120)
    name_planet = StringVar(value="HR 8799 b")
    T_star = DoubleVar() ; lg_star = DoubleVar() ; mag_star = DoubleVar()
    T_planet = DoubleVar() ; lg_planet = DoubleVar() ; mag_planet = DoubleVar(value=15.5)
    band0 = StringVar() ; separation_planet = DoubleVar()
    start_FC = BooleanVar(value=False)
    
    #---------------------------------------------------------------------------------
    
    # CONFIGURATION 
    frame1 = LabelFrame(fenetre, text="Observation setup", font='Helvetica 15 bold')
    frame1.pack(fill="both", expand="yes")
    
    #OPTIONS
    frame1_options = under_frame()
    frame1_options.instrument_options()
    
    #INSTRUMENTS
    frame1_instruments = LabelFrame(frame1, text="Instruments", font='Helvetica 10 bold')
    frame1_instruments.grid(column=0, row=0, padx=366)
    Label(frame1_instruments, text="Molecular Mapping :", relief=GROOVE,font='Helvetica 8 bold', padx=10, pady=10).grid(column=0, row=0)
    Radiobutton(frame1_instruments,text='HARMONI', value='HARMONI', variable=instrument,command=frame1_options.instrument_options).grid(column=0, row=1)
    Radiobutton(frame1_instruments,text='ERIS', value='ERIS', variable=instrument,command=frame1_options.instrument_options).grid(column=0, row=2)
    Radiobutton(frame1_instruments,text='MIRI/MRS', value='MIRIMRS', variable=instrument,command=frame1_options.instrument_options).grid(column=0, row=3)
    Label(frame1_instruments, text="", pady=3.5).grid(column=1, row=4)
    
    Label(frame1_instruments, text="ADI+RDI :", relief=GROOVE,font='Helvetica 8 bold', padx=10, pady=10).grid(column=1, row=0)
    Radiobutton(frame1_instruments,text='NIRCam', value='NIRCam', variable=instrument,command=frame1_options.instrument_options).grid(column=1, row=1)
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # CHOIX DU CALCUL  
    frame2 = LabelFrame(fenetre, text="Calculation parameters", font='Helvetica 15 bold')
    frame2.pack(fill="both", expand="yes")
    
    frame2_calculation = under_frame()
    frame2_calculation.calculation_choice()
    
    frame2_exposure_time = under_frame()
    frame2_exposure_time.exposure_time_choice() ; frame2_exposure_time.enter_exposure_time()
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # TARGET PARAMETRES  
    frame3 = LabelFrame(fenetre, text="Target parameters", font='Helvetica 15 bold',padx=105)
    frame3.pack(fill="both", expand="yes")
    
    
    frame3_T_star = under_frame()
    frame3_T_star.T_star_choice() ; frame3_T_star.enter_T_star()
    
    frame3_lg_star = under_frame()
    frame3_lg_star.lg_star_choice() ; frame3_lg_star.enter_lg_star()
    
    frame3_mag_star = under_frame()
    frame3_mag_star.mag_star_choice() ; frame3_mag_star.enter_mag_star()
    
    frame3_T_planet = under_frame()
    frame3_T_planet.T_planet_choice() ; frame3_T_planet.enter_T_planet()
    
    frame3_lg_planet = under_frame()
    frame3_lg_planet.lg_planet_choice() ; frame3_lg_planet.enter_lg_planet()
    
    frame3_mag_planet = under_frame()
    
    frame3_band0 = under_frame() ; frame3_name_planet = None
    frame3_band0.band0_choice() ; frame3_band0.enter_band0()
    
    frame3_separation_planet = under_frame()
    frame3_separation_planet.separation_planet_choice() ; frame3_separation_planet.enter_separation_planet()
    
    frame3_name_planet = under_frame()
    frame3_name_planet.name_planet_choice() ; frame3_name_planet.enter_name_planet()
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # LANCEMENT + FERMETURE FENETRE
    Button(fenetre,text="Start", font='Helvetica 20 bold',bg="dark orange",fg="black",command=lambda *args: start(fenetre,start_FC),pady=5,padx=5).pack()
    fenetre.mainloop()
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # ON RECUPERE LES VARIABLES
    instrument = instrument.get()
    calculation = calculation.get()
    coronagraph = coronagraph.get()
    if coronagraph == "None":
        coronagraph = None
    apodizer = apodizer.get()
    strehl = strehl.get()
    plot_mag = plot_mag.get()
    channel = channel.get()
    exposure_time = exposure_time.get()
    name_planet = name_planet.get()
    T_star = T_star.get() ; lg_star = lg_star.get() ; mag_star = mag_star.get()
    T_planet = T_planet.get() ; lg_planet = lg_planet.get() ; mag_planet = mag_planet.get()
    band0 = band0.get()
    try :
        separation_planet = separation_planet.get()
    except :
        separation_planet = None
    if separation_planet==0:
        separation_planet = None
    if name_planet=="" or name_planet=="None" or name_planet=="none" or name_planet=="None " or name_planet=="none " or name_planet=="NONE" or name_planet=="NONE ":
        name_planet = None
    if start_FC.get() :
        print("You have chosen the instrument : "+instrument)
        if instrument=="HARMONI":
            harmoni(calculation,T_planet,lg_planet,mag_star,band0,T_star,lg_star,exposure_time,apodizer=apodizer,strehl=strehl,model="BT-Settl",mag_planet=mag_planet,separation_planet=separation_planet ,name_planet=name_planet,plot_mag=plot_mag,return_SNR_planet=False)
        elif instrument=="ERIS":
            eris(calculation,T_planet,lg_planet,mag_star,band0,T_star,lg_star,exposure_time,model="BT-Settl",mag_planet=mag_planet,separation_planet=separation_planet ,name_planet=name_planet,plot_mag=plot_mag,return_SNR_planet=False)
        elif instrument=="MIRIMRS":
            mirimrs(calculation,T_planet,lg_planet,mag_star,band0,T_star,lg_star,exposure_time,model="BT-Settl",mag_planet=mag_planet,separation_planet=separation_planet ,name_planet=name_planet,channel=channel,plot_mag=plot_mag,return_SNR_planet=False)
        elif instrument=="NIRCam":
            nircam(calculation,T_planet,lg_planet,mag_star,band0,T_star,lg_star,exposure_time,model="BT-Settl",mag_planet=mag_planet,separation_planet=separation_planet ,name_planet=name_planet,plot_mag=plot_mag,return_SNR_planet=False)    
            
        
    
    
