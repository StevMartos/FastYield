import sys
import numpy as np
from sklearn.decomposition import PCA
from math import *
import tkinter as tk
from tkinter.ttk import *
from tkinter import filedialog, scrolledtext, messagebox
from PIL import ImageTk
from PIL.Image import fromarray, LANCZOS
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable

sys.path.insert(0, '../src')
from src.molecular_mapping import *
from src.FastCurves import *
from src.FastCurves_interface import *
from src.colormaps import *
from src.FastYield import *


class MyWindow(tk.Tk):  # https://koor.fr/Python/Tutoriel_Scipy_Stack/matplotlib_integration_ihm.wp

    def __init__(self):
        super().__init__()

        self.fastyield_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.filename_param = os.path.join(self.fastyield_dir, 'data_analysis', 'saved_parameters.txt')

        self.current_instru = None
        self.filename = ''
        self.width_image, self.height_image = 350, 350

        self.title("DataAnalysis")

        #####################################################FRAME LOAD FITS###########################################
        self.frameFITS = Frame(self, borderwidth=2, relief="groove")

        ##Loading fits
        labelFits = Label(self.frameFITS, text="FITS location:")
        labelFits.grid(row=0, column=0, padx=5, pady=5)

        self.entryFits = Entry(self.frameFITS, bd=5, width=100)
        self.entryFits.grid(row=0, column=1, padx=5, pady=5)

        browseButton = Button(self.frameFITS, text='Browse', command=self.browse_file)
        browseButton.grid(row=0, column=2, padx=5, pady=5)

        # Checkbutton autofill
        self.var_autofill = BooleanVar()
        self.c_autofill = Checkbutton(self.frameFITS, text="autofill", variable=self.var_autofill,
                                      command=self.autofillCall)
        self.c_autofill.grid(row=1, column=0, padx=5, pady=5)

        self.frameFITS.pack(side="top", fill='x')  #grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        ###################################################### MAIN FRAME LEFT ###########

        self.mainFrameLeft = ttk.Frame(self)

        ##################################################### FRAME INSTRUMENT ###########################################
        self.frameInstru = Frame(self.mainFrameLeft, borderwidth=2, relief="groove")
        #titleFrameInstru = Label(self.mainFrameLeft, text="Frame 1", font=("Helvetica", 12, "bold"))

        # CHOIX DE L'INSTRUMENT

        label_instru = Label(self.frameInstru, text="Instrument choice:")
        label_instru.grid(row=0, column=0, padx=5, pady=5)

        self.instruSelect = StringVar()
        self.stockInstru = ('ERIS/SPIFFIER', 'JWST/NIRSPEC', 'JWST/MIRI')
        self.listeInstru = Combobox(self.frameInstru, textvariable=self.instruSelect, values=self.stockInstru,
                                    state='readonly')
        self.listeInstru.set('ERIS/SPIFFIER')
        self.current_instru = self.listeInstru.get()
        self.listeInstru.bind("<<ComboboxSelected>>", self.selectInstruCall)
        self.listeInstru.grid(row=0, column=1, padx=5, pady=5)  #placement de la liste instrus

        # CHOIX DE LA BANDE

        label_band = Label(self.frameInstru, text="Band choice:")
        label_band.grid(row=1, column=0, padx=5, pady=5)

        self.bandSelect = StringVar()
        self.stockBand = ''
        self.listeBand = Combobox(self.frameInstru, textvariable=self.bandSelect, values=self.stockBand,
                                  state='readonly')
        self.listeBand.set('-----')
        self.listeBand.grid(row=1, column=1, padx=5, pady=5)  # placement de la liste bandes instru

        self.frameInstru.pack(side="top")  #grid(row=1, column=0, padx=5, pady=5)

        ##############################FRAME DATA##################################
        self.frameData = Frame(self.mainFrameLeft, borderwidth=2, relief="groove")

        # Parameters entry
        label_magnitude = Label(self.frameData, text="star magnitude:")
        label_magnitude.grid(row=0, column=0, padx=5, pady=5)
        self.entryMagnitude = Entry(self.frameData, bd=5)
        self.entryMagnitude.grid(row=0, column=1, padx=5, pady=5)

        self.bandStarSelect = StringVar()
        self.stockBandStar = ("J", "H", "Ks", "K", "L", "L'", "instru", "NIR")
        self.listeBandStar = Combobox(self.frameData, textvariable=self.bandStarSelect, values=self.stockBandStar,
                                      state='readonly', width=6)
        self.listeBandStar.set('-----')
        self.listeBandStar.grid(row=0, column=2, padx=5, pady=5)

        label_DIT = Label(self.frameData, text="DIT (sec):")
        label_DIT.grid(row=1, column=0, padx=5, pady=5)
        self.entryDIT = Entry(self.frameData, bd=5)
        self.entryDIT.grid(row=1, column=1, padx=5, pady=5)

        label_NDIT = Label(self.frameData, text="NDIT:")
        label_NDIT.grid(row=2, column=0, padx=5, pady=5)
        self.entryNDIT = Entry(self.frameData, bd=5)
        self.entryNDIT.grid(row=2, column=1, padx=5, pady=5)

        label_NINT = Label(self.frameData, text="NINT:")
        label_NINT.grid(row=3, column=0, padx=5, pady=5)
        self.entryNINT = Entry(self.frameData, bd=5)
        self.entryNINT.grid(row=3, column=1, padx=5, pady=5)

        label_pixelscale = Label(self.frameData, text="pixelscale (mas):")
        label_pixelscale.grid(row=4, column=0, padx=5, pady=5)
        self.entryPixelscale = Entry(self.frameData, bd=5)
        self.entryPixelscale.grid(row=4, column=1, padx=5, pady=5)

        label_rho_max = Label(self.frameData, text="rho max:")
        label_rho_max.grid(row=5, column=0, padx=5, pady=5)
        self.entry_rho_max = Entry(self.frameData, bd=5)
        self.entry_rho_max.grid(row=5, column=1, padx=5, pady=5)

        # Entry unit
        labelUnit = Label(self.frameData, text="Data unit:")
        labelUnit.grid(row=6, column=0, padx=5, pady=5)
        self.unitSelect = StringVar()
        self.stockUnit = ('-----', 'MJy/sr', 'ADU', 'e-')
        self.listeUnit = Combobox(self.frameData, textvariable=self.unitSelect, values=self.stockUnit,
                                  state='readonly', width=6)
        self.listeUnit.grid(row=6, column=1, padx=5, pady=5)

        # Pixsteradian
        label_Pixsteradian = Label(self.frameData, text="Pixel steradian:")
        label_Pixsteradian.grid(row=7, column=0, padx=5, pady=5)
        self.entryPixsteradian = Entry(self.frameData, bd=5)
        self.entryPixsteradian.grid(row=7, column=1, padx=5, pady=5)

        # collapse case
        label_collapse = Label(self.frameData, text="Cube collapsing method:")
        label_collapse.grid(row=8, column=0, padx=5, pady=5)

        self.collapseModeSelect = StringVar()
        self.stockMode = ('Mean', 'Sum')
        self.listeMode = Combobox(self.frameData, textvariable=self.collapseModeSelect, values=self.stockMode,
                                  state='readonly')
        self.listeMode.set('Mean')
        self.listeMode.grid(row=8, column=1, padx=5, pady=5)  # placement de la liste instrus

        labelLmin = Label(self.frameData, text="$λ_{min}$(µm):")
        labelLmin.grid(row=9, column=0, padx=5, pady=5)
        self.entryLmin = Entry(self.frameData, bd=5)
        self.entryLmin.grid(row=9, column=1, padx=5, pady=5)

        labelLmax = Label(self.frameData, text="$λ_{max}$(µm):")
        labelLmax.grid(row=10, column=0, padx=5, pady=5)
        self.entryLmax = Entry(self.frameData, bd=5)
        self.entryLmax.grid(row=10, column=1, padx=5, pady=5)

        labelDlambda = Label(self.frameData, text="Δλ(µm):")
        labelDlambda.grid(row=11, column=0, padx=5, pady=5)
        self.entryDlambda = Entry(self.frameData, bd=5)
        self.entryDlambda.grid(row=11, column=1, padx=5, pady=5)

        self.frameData.pack(side="top")  #grid(row=2, column=0, padx=5, pady=5)

        ###################FRAME LOG############################
        self.frameLog = Frame(self, borderwidth=2, relief="groove")

        self.log_text = scrolledtext.ScrolledText(self.frameLog, width=100, height=2)
        self.log_text.grid(row=0, column=0, padx=5, pady=5)

        self.frameLog.pack(side="bottom", fill='x')  # grid(row=1, column=1, padx=5, pady=5)

        ##############################FRAME PARAM MM###########################
        self.frameParamMM = Frame(self.mainFrameLeft, borderwidth=2, relief="groove")

        ## Param Molecular Mapping
        labelTeff = Label(self.frameParamMM, text="$T_{eff}$ (K):")
        labelTeff.grid(row=0, column=0, padx=5, pady=5)

        self.entryTeff = Entry(self.frameParamMM, bd=5)
        self.entryTeff.insert(0, "1600")
        self.entryTeff.grid(row=0, column=1, padx=5, pady=5)

        labelModel = Label(self.frameParamMM, text="Template model:")
        labelModel.grid(row=1, column=0, padx=5, pady=5)

        self.selectModel = StringVar()
        self.stockModel = ("BT-Settl", "BT-Dusty", "Exo-REM", "PICASO", "Morley", "Saumon", "SONORA")
        self.listeModel = Combobox(self.frameParamMM, textvariable=self.selectModel, values=self.stockModel,
                                   state='readonly')
        self.listeModel.set("BT-Settl")
        self.listeModel.grid(row=1, column=1, padx=5, pady=5)

        labelPCA = Label(self.frameParamMM, text="Ncomponents for PCA subtraction:")
        labelPCA.grid(row=2, column=0, padx=5, pady=5)

        self.entryPCA = Entry(self.frameParamMM, bd=5)
        self.entryPCA.grid(row=2, column=1, padx=5, pady=5)
        self.entryPCA.insert(0, "0")

        self.frameParamMM.pack(side="top")  # grid(row=2, column=0, padx=5, pady=5)
        self.mainFrameLeft.pack(side='left')

        ##############################SECOND FRAME LEFT########################
        self.secondFrameLeft = Frame(self, borderwidth=2, relief="groove")
        ##############################FRAME MM##################################
        self.frameMM = Frame(self.secondFrameLeft, borderwidth=2, relief="groove")
        ## image du cube
        self.canvas = tk.Canvas(self.frameMM, width=self.width_image, height=self.height_image)
        self.canvas.grid(row=0, column=0, padx=5, pady=5)

        ## bouton Molecular Mapping
        self.launchMM = Button(self.frameMM, text='Launch MM', command=self.launchMMCall)
        self.launchMM.grid(row=0, column=1, padx=5, pady=5)

        ## image du MM
        self.canvas_mm_parent = tk.Canvas(self.frameMM, width=self.width_image, height=self.height_image)
        self.canvas_mm_parent.grid(row=0, column=2, padx=5, pady=5)

        ## bouton launch analysis Molecular Mapping
        self.launchMM = Button(self.frameMM, text='Source characterization', command=self.launchAnalysisMMCall)
        self.launchMM.grid(row=0, column=3, padx=5, pady=5)

        self.frameMM.pack(side='top')  #grid(row=1, column=1, padx=5, pady=5)

        #####################FRAME NOISE#######################
        self.frameNoise = Frame(self.secondFrameLeft, borderwidth=2, relief="groove")

        self.buttonStatics = Button(self.frameNoise, text='Estimate Noise', command=self.launchNoiseCurvesCall)
        self.buttonStatics.grid(row=0, column=0, padx=5, pady=5)

        self.canvas_noise = tk.Canvas(self.frameNoise, width=self.width_image, height=self.height_image)
        self.canvas_noise.grid(row=0, column=1, padx=5, pady=5)

        self.frameNoise.pack()
        self.secondFrameLeft.pack(side='left')

        self.rowconfigure(0, weight=1)  # Permet à la première ligne de s'étirer
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.columnconfigure(0, weight=1)  # Permet à la première colonne de s'étirer
        self.columnconfigure(1, weight=1)  # Permet à la deuxième colonne de s'étirer
        self.columnconfigure(2, weight=1)

        try:
            self.loadParam(self.filename_param)
        except Exception as e:
            self.log_message("[LOG] Could not load previous parameters")

    def selectInstruCall(self, event):
        self.current_instru = self.instruSelect.get()

        if self.current_instru == 'ERIS/SPIFFIER':
            self.stockBand = (
                "J_low", "H_low", "K_low", "J_short", "J_middle", "J_long", "H_short", "H_middle", "H_long", "K_short",
                "K_middle", "K_long")
        elif self.current_instru == 'JWST/MIRI':
            self.stockBand = ("Channel1", "Channel2", "1SHORT", "1MEDIUM", "1LONG", "2SHORT", "2MEDIUM", "2LONG")
        elif self.current_instru == 'JWST/NIRSPEC':
            self.stockBand = (
                "G140M_F070LP", "G140M_F100LP", "G235M_F170LP", "G395M_F290LP", "G140H_F070LP", "G140H_F100LP",
                "G235H_F170LP", "G395H_F290LP")
        self.listeBand.pack_forget()
        self.listeBand = Combobox(self.frameInstru, textvariable=self.bandSelect, values=self.stockBand,
                                  state='readonly')
        self.listeBand.grid(row=1, column=1, padx=5, pady=5)

    def browse_file(self):
        filetypes = (("FITS files", "*.fits"),)
        self.filename = filedialog.askopenfilename(filetypes=filetypes)
        if self.filename:
            self.cube_data, _ = crop(fits.getdata(self.filename), fits.getdata(self.filename))
            if len(self.cube_data.shape) == 3:
                # Convertir l'image FITS en une image PIL
                N_lambda = self.cube_data.shape[0]
                image = fromarray(self.cube_data[N_lambda // 2])

                image_resized = image.resize((self.width_image, self.height_image), LANCZOS)

                # Convertir l'image PIL en format Tkinter PhotoImage
                photo = ImageTk.PhotoImage(image_resized)

                self.entryFits.delete(0, END)
                self.entryFits.insert(0, self.filename)

                # Afficher l'image sur le Canvas
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo
            else:
                self.log_message("ERROR: Wrong dimension for the cube file")

    def launchMMCall(self):
        perform = 0

        if self.current_instru == "JWST/MIRI":
            instru = "MIRIMRS"
            self.tellurics = False
        elif self.current_instru == "ERIS/SPIFFIER":
            instru = "ERIS"
            self.tellurics = True
        elif self.current_instru == "JWST/NIRSPEC":
            instru = "NIRSpec"
            self.tellurics = False

        self.Band = self.listeBand.get()
        self.T = int(self.entryTeff.get())
        self.Model = self.selectModel.get()
        self.Lmin = self.entryLmin.get()
        self.Lmax = self.entryLmax.get()
        self.Dlambda = self.entryDlambda.get()
        self.pxsteradian = self.entryPixsteradian.get()

        if any(x == '' for x in (self.Band, self.T, self.Model, self.Lmin, self.Lmax, self.Dlambda, self.pxsteradian)):
            self.log_message("ERROR: At least one entry has not been filled in")
        else:
            try:
                self.Lmin = float(self.Lmin)
                self.Lmax = float(self.Lmax)
                self.Dlambda = float(self.Dlambda)
                self.Wavelength = np.linspace(self.Lmin, self.Lmax, num=self.cube_data.shape[0])
                config_data = get_config_data(instru)
                self.R = config_data['gratings'][self.Band].R
                self.pxsteradian = float(self.entryPixsteradian.get())
                self.ConvertUnitData()
                try:
                    self.trans = transmission(instru, self.Wavelength, self.Band, self.tellurics, None)
                    self.trans[np.isnan(self.trans)] = 0
                    self.cube_data_hf, _ = stellar_high_filtering(np.copy(self.cube_data), False, self.R, 100,
                                                                  "gaussian",
                                                                  print_value=False)
                    if self.entryPCA.get() != "0":
                        self.PCA_subtraction(int(self.entryPCA.get()))

                    self.correlation_map, self.rv, _ = molecular_mapping_rv(instru, self.Band, self.cube_data_hf,
                                                                            self.T, 4.0,
                                                                            self.Model, self.Wavelength,
                                                                            self.trans,
                                                                            "SNR_rv",
                                                                            self.R, 100, "gaussian", broad=0, rv=0,
                                                                            print_value=True)
                    self.log_message("[LOG]: Molecular Mapping performed successfully")
                    perform = 1

                except Exception as e:
                    print(e)
                    self.log_message("ERROR: failed to perform molecular mapping on the data")
                    self.log_message(str(e))

            except Exception as e:
                self.log_message("ERROR: type for Lmin and/or Lmax and/or Dlambda is wrong")
                self.log_message(str(e))

        if perform:
            try:
                self.plot_correlation_map()
                self.log_message("[LOG]: Correlation Maps successfully plotted")
            except Exception as e:
                self.log_message("ERROR: failed to plot the correlation map")
                self.log_message(str(e))

    def normalize_correlation_map(self):

        self.normalized_map = np.zeros_like(self.correlation_map)
        _, nbLine, nbColumn = self.cube_data_hf.shape

        for k in range(0, self.correlation_map.shape[0]):
            for i in range(1, nbLine // 2):
                mask = annular_mask(i, i + 1, value=np.nan, size=(nbLine, nbColumn))
                ccfMasked = np.copy(self.correlation_map[k]) * mask
                std = np.nanstd(ccfMasked, axis=(0, 1))
                ccfMasked /= std
                ccfMasked[np.isnan(ccfMasked)] = 0
                self.normalized_map[k] += ccfMasked

    def plot_correlation_map(self):

        if 1 == 0:
            self.normalize_correlation_map()
            self.mm_map = np.copy(self.normalized_map)
        else:
            self.mm_map = np.copy(self.correlation_map)

        c_min = np.nanmin(self.mm_map)
        c_max = np.nanmax(self.mm_map)
        r = (c_max - c_min) / 100

        self.fig_mm = Figure(figsize=(self.width_image / 100, self.height_image / 100))
        self.ax_mm = self.fig_mm.add_subplot(111)
        self.ax_mm.imshow(self.mm_map[0], cmap='viridis')
        self.ax_mm.set_title(f"Correlation map for rv= {round(self.rv[0])} km/s")
        self.canvas_mm = FigureCanvasTkAgg(self.fig_mm, master=self.canvas_mm_parent)
        self.canvas_mm.draw()
        self.canvas_mm.get_tk_widget().grid(row=0, column=2, padx=5, pady=5)

        self.slice_slider_vmax = tk.Scale(self.frameMM, label="vmax", from_=c_min, to=c_max, resolution=r,
                                          orient=tk.HORIZONTAL,
                                          command=self.update_plot_correlation_map)
        self.slice_slider_vmax.set(c_max)
        self.slice_slider_vmax.grid(row=1, column=2, padx=5, pady=5)

        self.slice_slider_vmin = tk.Scale(self.frameMM, label="vmin", from_=c_min, to=c_max, resolution=r,
                                          orient=tk.HORIZONTAL,
                                          command=self.update_plot_correlation_map)
        self.slice_slider_vmin.set(c_min)
        self.slice_slider_vmin.grid(row=2, column=2, padx=5, pady=5)

        self.slice_slider_rv = ttk.Scale(self.frameMM, from_=0, to=self.correlation_map.shape[0] - 1,
                                         orient=tk.HORIZONTAL, command=self.update_plot_correlation_map)
        self.slice_slider_rv.grid(row=3, column=2, padx=5, pady=5)

        self.toolbarFrame = Frame(master=self.frameMM)
        self.toolbarFrame.grid(row=4, column=2)  #TODO change
        self.toolbar = NavigationToolbar2Tk(self.canvas_mm, self.toolbarFrame)

    def update_plot_correlation_map(self, event=None):
        self.vmax = self.slice_slider_vmax.get()
        self.vmin = self.slice_slider_vmin.get()
        self.current_slice = int(self.slice_slider_rv.get())
        slice_data = self.mm_map[self.current_slice, :, :]
        self.ax_mm.clear()
        self.ax_mm.imshow(slice_data, cmap='viridis', vmin=self.vmin, vmax=self.vmax)
        self.ax_mm.set_title(f"Correlation map for rv= {round(self.rv[self.current_slice])} km/s")
        self.canvas_mm.draw()

    def autofillCall(self):
        self.entryNINT.delete(0, tk.END)
        self.entryNDIT.delete(0, tk.END)
        self.entryDIT.delete(0, tk.END)
        self.entryLmin.delete(0, tk.END)
        self.entryLmax.delete(0, tk.END)
        self.entryPixelscale.delete(0, tk.END)
        self.entryDlambda.delete(0, tk.END)
        self.entryPixsteradian.delete(0, tk.END)

        if self.filename == '':
            self.var_autofill.set(False)
            self.log_message("ERROR: you must load a fits file before")
        else:
            if self.var_autofill.get():
                data = fits.open(self.filename)

                if self.current_instru == "ERIS/SPIFFIER":
                    header = data[0].header
                    self.log_message("[LOG] : Reading header fits for autofill")
                    self.autoBand = header['HIERARCH ESO INS3 SPGW NAME']
                    self.autoNINT = "Not defined"
                    self.autoNDIT = "Not defined"
                    self.autoDIT = float(header["EXPTIME"])
                    self.autoLmin = float(header["CRVAL3"])
                    self.autoDlambda = float(header["CDELT3"])
                    self.autoLmax = self.autoLmin + (data[0].data.shape[0] - 1) * self.autoDlambda
                    self.autoPixelscale = np.abs(float(header["CD1_1"])) * 3.6e6
                    self.autoPxsteradian = "Not defined"
                    self.gain = header['HIERARCH ESO DET CHIP GAIN']
                    self.autoUnit = "-----"
                    self.targetInformation = f"Name of the target:{header['HIERARCH ESO OBS TARG NAME']}"

                elif self.current_instru == "JWST/MIRI":
                    header1 = data[0].header
                    header2 = data[1].header
                    self.log_message("[LOG] : Reading header fits for autofill")
                    self.autoBand = f"Channel{header1['CHANNEL']}"  #f"{header1['CHANNEL']}SHORT"
                    self.autoNINT = int(header1["NINTS"])
                    self.autoNDIT = int(header1["NGROUPS"]) * int(header1["NFRAMES"])
                    self.autoDIT = float(header1["TFRAME"])
                    self.autoLmin = float(header2["CRVAL3"])
                    self.autoDlambda = float(header2["CDELT3"])
                    self.autoLmax = self.autoLmin + (data[1].data.shape[0] - 1) * self.autoDlambda
                    self.autoPixelscale = float(header2["CDELT1"]) * 3.6e6
                    self.autoPxsteradian = float(header2['PIXAR_SR'])
                    self.autoUnit = header2["BUNIT"]
                    self.targetInformation = f"Proper name of the target:{header1['TARGPROP']} \n Standard astronomical catalog name for tar: {header1['TARGNAME']}"
                    targetCategory = header1['TARGCAT']
                    if targetCategory != 'Star':
                        self.log_message("Warning: The target category from the data is not 'Star'")

                elif self.current_instru == "JWST/NIRSPEC":
                    header1 = data[0].header
                    header2 = data[1].header
                    self.log_message("[LOG] : Reading header fits for autofill")
                    self.autoBand = f"{header1['GRATING']}_{header1['FILTER']}"
                    self.autoNINT = int(header1["NINTS"])
                    self.autoNDIT = int(header1["NGROUPS"]) * int(header1["NFRAMES"])
                    self.autoDIT = float(header1["TFRAME"])
                    self.autoLmin = float(header2["CRVAL3"])
                    self.autoDlambda = float(header2["CDELT3"])
                    self.autoLmax = self.autoLmin + (data[1].data.shape[0] - 1) * self.autoDlambda
                    self.autoPixelscale = float(header2["CDELT1"]) * 3.6e6
                    self.autoPxsteradian = float(header2['PIXAR_SR'])
                    self.autoUnit = header2["BUNIT"]
                    self.targetInformation = f"Proper name of the target:{header1['TARGPROP']} \n Standard astronomical catalog name for tar: {header1['TARGNAME']}"
                    targetCategory = header1['TARGCAT']
                    if targetCategory != 'Star':
                        self.log_message("Warning: The target category from the data is not 'Star'")

                self.listeBand.set(self.autoBand)
                self.entryNINT.insert(0, str(self.autoNINT))
                self.entryNDIT.insert(0, str(self.autoNDIT))
                self.entryDIT.insert(0, str(self.autoDIT))
                self.entryLmin.insert(0, str(self.autoLmin))
                self.entryLmax.insert(0, str(self.autoLmax))
                self.entryDlambda.insert(0, str(self.autoDlambda))
                self.entryPixelscale.insert(0, str(self.autoPixelscale))
                self.entryPixsteradian.insert(0, str(self.autoPxsteradian))
                self.listeUnit.set(self.autoUnit)
                self.log_message(self.targetInformation)


            else:

                self.entryNDIT.insert(0, "")
                self.entryDIT.insert(0, "")
                self.entryLmin.insert(0, "")
                self.entryLmax.insert(0, "")
                self.entryDlambda.insert(0, "")

    def ConvertUnitData(self):
        self.cube_data, _ = crop(fits.getdata(self.filename), fits.getdata(self.filename))
        try:
            self.exposure_time = int(self.entryNINT.get()) * int(self.entryNDIT.get()) * float(self.entryDIT.get())
            current_unit = self.listeUnit.get()
            if current_unit == "MJy/sr":
                config_data = get_config_data(self.instruTranslation(self.listeInstru.get()))
                area = config_data['telescope']['area']
                for i, lamb in enumerate(self.Wavelength):
                    self.cube_data[i] *= self.pxsteradian * 1e6 * 1e-26 * float(self.Dlambda) * 1e-6 * lamb * 1e-6 / (
                            h * c) * area * self.exposure_time * c / ((lamb * 1e-6) ** 2)
                self.log_message("[LOG]: Successfully converted the data cube in photon/pixel")

            elif current_unit == "ADU":
                self.cube_data *= self.gain
                if self.listeMode.get() == 'MEAN':
                    self.cube_data *= self.exposure_time
                self.log_message("[LOG]: Successfully converted the data cube in photon/pixel")

            elif current_unit == "e-":
                "DO NOTHING"
                self.log_message("[LOG]: Successfully converted the data cube in photon/pixel")

            else:
                self.log_message("ERROR: Unknown unit for data")


        except Exception as e:
            self.log_message("ERROR: failed to convert the data cube in photon/pixel")
            self.log_message(str(e))

    def launchNoiseCurvesCall(self):
        self.saveTempParam()
        #FastCurves prediction
        self.Band = self.listeBand.get()
        self.T = int(self.entryTeff.get())
        self.Model = self.selectModel.get()
        self.Lmin = self.entryLmin.get()
        self.Lmax = self.entryLmax.get()
        self.Dlambda = self.entryDlambda.get()
        self.Lmin = float(self.Lmin)
        self.Lmax = float(self.Lmax)
        self.Dlambda = float(self.Dlambda)
        self.Wavelength = np.linspace(self.Lmin, self.Lmax, num=self.cube_data.shape[0])#np.arange(self.Lmin, self.Lmax, self.Dlambda)
        config_data = get_config_data(self.instruTranslation(self.listeInstru.get()))
        self.R = config_data['gratings'][self.Band].R

        self.DIT = float(self.entryDIT.get())
        self.NDIT = int(self.entryNDIT.get())
        self.NINT = int(self.entryNINT.get())
        self.exposure_time = self.DIT * self.NDIT * self.NINT
        self.magStar = float(self.entryMagnitude.get())
        self.bandMagStar = self.listeBandStar.get()

        try:
            planet_spectrum = load_planet_spectrum(self.T, 4.0, self.Model)
            star_spectrum = load_star_spectrum(4000, 4.0)  #np.nansum(self.cube_data, (1, 2))

            nameBand, separationBand, noiseBand, NDIT_FC = FastCurves("Noise",
                                                                      self.instruTranslation(self.listeInstru.get()),
                                                                      self.exposure_time / 60, None, self.magStar,
                                                                      self.bandMagStar, planet_spectrum, star_spectrum,
                                                                      self.tellurics,
                                                                      apodizer=None, strehl=None, coronagraph=None,
                                                                      plot_mag=False,
                                                                      separation_planet=None, mag_planet=None,
                                                                      channel=False, systematic=False, show_plot=False,
                                                                      print_value=False,
                                                                      post_processing="molecular mapping",
                                                                      sep_unit="arcsec", bkgd=None,
                                                                      star_pos="center", Rc=100,
                                                                      band_only=None, used_filter="gaussian")

            for i, name in enumerate(nameBand):
                if name == self.listeBand.get():
                    self.separation_FastCurves = separationBand[i] * 1000
                    self.noise_FastCurves = np.sqrt(noiseBand[i] * NDIT_FC)
            print(self.noise_FastCurves)

            self.log_message("[LOG]: Noise level successfully estimated with FastCurves")
        except Exception as e:
            self.log_message("ERROR: failed to estimate the noise level with FastCurves")
            self.log_message(str(e))

        try:
            #Noise estimated in the data
            N = int(float(self.entry_rho_max.get()) // float(self.entryPixelscale.get()))
            print(N)
            if N <= 1:
                N = 10
                self.entry_rho_max.delete(0, tk.END)
                self.entry_rho_max.insert(0, N * float(self.entryPixelscale.get()))
                self.log_message("[LOG]: Maximal separation set to 10 times the pixelscale")

            self.noise_est = np.zeros((2, N))
            Nl = self.correlation_map.shape[0]

            for i in range(1, N):
                _, nbLine, nbColumn = self.cube_data_hf.shape
                mask = annular_mask(i, i + 1, value=np.nan, size=(nbLine, nbColumn))
                ccfMasked = np.copy(self.correlation_map[Nl // 2]) * mask
                self.noise_est[0, i] = i * float(self.entryPixelscale.get())
                self.noise_est[1, i] = np.nanstd(ccfMasked, axis=(0, 1))

            self.log_message("[LOG]: Noise level from the data successfully estimated")
        except Exception as e:
            self.log_message("ERROR: failed to estimate the noise level from the data")
            self.log_message(str(e))

        self.plotNoiseAnalysis()

    def plotNoiseAnalysis(self):
        try:
            fig, ax = plt.subplots(figsize=((self.width_image + 200) / 100, self.height_image / 100))

            print(self.noise_est[1])

            ax.plot(self.separation_FastCurves, self.noise_FastCurves, label='Noise level estimated with FastCurves')
            ax.plot(self.noise_est[0], self.noise_est[1], label='Noise level estimated in the data')
            ax.legend()
            ax.set_xlabel('Separation (mas)')
            ax.set_ylabel('Noise Level (e-)')
            ax.set_title('Noise level wrt the separation')

            canvas = FigureCanvasTkAgg(fig, master=self.canvas_noise)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)
            self.log_message("[LOG]: Noise level successfully ploted")

        except Exception as e:
            self.log_message("ERROR: failed to plot the noise level")
            self.log_message(str(e))

    def PCA_subtraction(self, n_comp_sub):
        pca = PCA(n_components=n_comp_sub)

        nbChannel, nbColumn, nbLine = self.cube_data_hf.shape
        X = np.reshape(self.cube_data_hf, (nbChannel, nbColumn * nbLine)).transpose()
        X[np.isnan(X)] = 0
        X_r = pca.fit_transform(X)
        X_restored = pca.inverse_transform(X_r)

        X_sub = (X - X_restored).transpose()
        X_sub = np.reshape(X_sub, (nbChannel, nbColumn, nbLine))

        self.cube_data_hf = X_sub

    def instruTranslation(self, nameInstru):
        if nameInstru == "JWST/MIRI":
            instru = "MIRIMRS"
        elif nameInstru == "ERIS/SPIFFIER":
            instru = "ERIS"
        elif nameInstru == "JWST/NIRSPEC":
            instru = "NIRSpec"
        else:
            instru = None

        return instru

    def saveTempParam(self):
        fits = self.entryFits.get()
        instru = self.listeInstru.get()
        band_instru = self.listeBand.get()
        star_mag = self.entryMagnitude.get()
        band_mag = self.listeBandStar.get()
        DIT = self.entryDIT.get()
        NDIT = self.entryNDIT.get()
        NINT = self.entryNINT.get()
        pixscale = self.entryPixelscale.get()
        rho_max = self.entry_rho_max.get()
        data_unit = self.listeUnit.get()
        pixsteradian = self.entryPixsteradian.get()
        cube_collapse = self.listeMode.get()
        lmin = self.entryLmin.get()
        lmax = self.entryLmax.get()
        dlamb = self.entryDlambda.get()
        Teff = self.entryTeff.get()
        model = self.listeModel.get()

        parameters = [fits, instru, band_instru, star_mag, band_mag, DIT, NDIT, NINT, pixscale, rho_max, data_unit,
                      pixsteradian, cube_collapse, lmin, lmax, dlamb, Teff, model]

        self.writeParam(self.filename_param, parameters)

    def writeParam(self, filename, parameters):
        with open(filename, 'w') as file:
            for param in parameters:
                file.write(param + '\n')

    def loadParam(self, filename):
        parameters = []
        with open(filename, 'r') as file:
            for line in file.readlines():
                parameters.append(line.strip())

        self.filename = parameters[0]

        self.entryFits.delete(0, tk.END)
        self.entryFits.insert(0, self.filename)
        self.cube_data, _ = crop(fits.getdata(self.filename), fits.getdata(self.filename))
        if len(self.cube_data.shape) == 3:
            # Convertir l'image FITS en une image PIL
            N_lambda = self.cube_data.shape[0]
            image = fromarray(self.cube_data[N_lambda // 2])

            image_resized = image.resize((self.width_image, self.height_image), LANCZOS)

            # Convertir l'image PIL en format Tkinter PhotoImage
            photo = ImageTk.PhotoImage(image_resized)

            self.entryFits.delete(0, END)
            self.entryFits.insert(0, self.filename)

            # Afficher l'image sur le Canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
        else:
            self.log_message("ERROR: Wrong dimension for the cube file")
        self.listeInstru.set(parameters[1])
        self.selectInstruCall(None)
        self.listeBand.set(parameters[2])
        self.entryMagnitude.delete(0, tk.END)
        self.entryMagnitude.insert(0, parameters[3])
        self.listeBandStar.set(parameters[4])
        self.entryDIT.delete(0, tk.END)
        self.entryDIT.insert(0, parameters[5])
        self.entryNDIT.delete(0, tk.END)
        self.entryNDIT.insert(0, parameters[6])
        self.entryNINT.delete(0, tk.END)
        self.entryNINT.insert(0, parameters[7])
        self.entryPixelscale.delete(0, tk.END)
        self.entryPixelscale.insert(0, parameters[8])
        self.entry_rho_max.delete(0, tk.END)
        self.entry_rho_max.insert(0, parameters[9])
        self.listeUnit.set(parameters[10])
        self.entryPixsteradian.delete(0, tk.END)
        self.entryPixsteradian.insert(0, parameters[11])
        self.listeMode.set(parameters[12])
        self.entryLmin.delete(0, tk.END)
        self.entryLmin.insert(0, parameters[13])
        self.entryLmax.delete(0, tk.END)
        self.entryLmax.insert(0, parameters[14])
        self.entryDlambda.delete(0, tk.END)
        self.entryDlambda.insert(0, parameters[15])
        self.entryTeff.delete(0, tk.END)
        self.entryTeff.insert(0, parameters[16])
        self.listeModel.set(parameters[17])

        return parameters

    def log_message(self, message):
        """
        Parameters
        ----------
        message (str) : message to add in the log
        -------
        """
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def launchAnalysisMMCall(self):

        self.windowMM = Toplevel(self)
        self.windowMM.title("Molecular mapping analysis")

        self.frameAnalysisMMparam = Frame(self.windowMM, borderwidth=2, relief="groove")

        labelX0 = Label(self.frameAnalysisMMparam, text="Source X0:")
        labelX0.grid(row=0, column=0, padx=5, pady=5)
        self.entryX0 = Entry(self.frameAnalysisMMparam, bd=5)
        self.entryX0.grid(row=0, column=1, padx=5, pady=5)

        labelY0 = Label(self.frameAnalysisMMparam, text="Source Y0:")
        labelY0.grid(row=1, column=0, padx=5, pady=5)
        self.entryY0 = Entry(self.frameAnalysisMMparam, bd=5)
        self.entryY0.grid(row=1, column=1, padx=5, pady=5)

        labelAperture = Label(self.frameAnalysisMMparam, text="Aperture size (pxl):")
        labelAperture.grid(row=2, column=0, padx=5, pady=5)
        self.entryAperture = Entry(self.frameAnalysisMMparam, bd=5)
        self.entryAperture.grid(row=2, column=1, padx=5, pady=5)

        self.canvas_first_mm = Canvas(self.frameAnalysisMMparam, width=self.width_image, height=self.height_image)
        self.canvas.grid(row=0, column=2, padx=5, pady=5)

        fig, axs = plt.subplots()
        axs.imshow(self.mm_map[self.mm_map.shape[0]//2])
        canvas = FigureCanvasTkAgg(fig, master=self.frameAnalysisMMparam)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=2)

        self.buttonLaunchMMgrid = Button(self.frameAnalysisMMparam, text='Plot Analysis', command=self.launchMMgrid)
        self.buttonLaunchMMgrid.grid(row=3, column=0)

        self.frameAnalysisMMparam.pack()

    def launchMMgrid(self):
        y0, x0 = int(self.entryY0.get()), int(self.entryX0.get())
        a = int(self.entryAperture.get())//2
        T = np.arange(800, 2900, 100)
        model = ['BT-Settl', 'Exo-REM']
        lg = np.arange(3.5, 5, 0.5)
        instru = self.instruTranslation(self.listeInstru.get())

        self.correlation = np.zeros((len(model), T.shape[0], lg.shape[0]))
        for k, m in enumerate(model):
            for i, t in enumerate(T):
                for j, g in enumerate(lg):
                    correlation_map, rv, _ = molecular_mapping_rv(instru, self.Band, self.cube_data_hf[:, y0-a:y0+a+1, x0-a:x0+a+1],
                                                                  t, g,
                                                                  m, self.Wavelength,
                                                                  self.trans,
                                                                  "SNR_rv",
                                                                  self.R, 100, "gaussian", broad=0, rv=0,
                                                                  print_value=True)

                    correlation_max = np.nanmax(correlation_map)
                    #rv_max = rv[np.where(correlation_map[:, y0, x0] == correlation_max)]
                    self.correlation[k, i, j] = correlation_max

        self.plot_characterization(T, model, lg)

    def plot_characterization(self, T, model, lg):

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.correlation[0])
        axs[0].set_title(f"{model[0]}")
        #axs[0].set_xticks(lg)
        #axs[0].set_yticks(T)
        axs[0].set_xlabel("log(g)")
        axs[0].set_ylabel("Teff")
        axs[1].imshow(self.correlation[1])
        axs[1].set_title(f"{model[1]}")
        #axs[1].set_xticks(lg)
        #axs[1].set_yticks(T)
        axs[1].set_xlabel("log(g)")
        axs[1].set_ylabel("Teff")

        #fig.tight_layout()

        # Convertir les figures en widgets Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frameAnalysisMMparam)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=2)




def DataAnalysis_interface():
    app = MyWindow()
    app.mainloop()
