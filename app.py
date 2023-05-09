import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient
import requests
import math
from streamlit_extras.chart_container import chart_container

import streamlit as st
import time
import copy


class Timeserier:
    def __init__(self):
        self.df = pd.DataFrame()
        
    def legg_inn_timeserie(self, timeserie, timeserie_navn):
        self.df[timeserie_navn] = timeserie

# ---------------------------------------------------------------------------------------------------------------
# -- Hjelpefunksjoner PVGIS
class BaseValidationError(ValueError):
    pass
class RoofValueListOrienteringNotEqualLengthError(BaseValidationError):
    pass
class RoofValueListArealerNotEqualLengthError(BaseValidationError):
    pass

class BaseValidationError(ValueError):
    pass

class InvalidMountingplaceValueError(BaseValidationError):
    pass

class Roof():
    def __init__(self, lat, lon, angle, aspect, footprint_area, loss = 14, mountingplace='free'):
        allowed_mountingplace= ['free', 'building']
        #if mountingplace not in allowed_mountingplace:
            #arcpy.AddMessage(f"""Mountingplace value {mountingplace} 
            #                 is not valid. Must be equal to
            #                 {allowed_mountingplace[0]}
            #                 or {allowed_mountingplace[1]}""")
            #raise InvalidMountingplaceValueError(mountingplace)
        self.mountingplace= mountingplace
        self.lat= lat
        self.lon= lon
        self.angle= angle
        self.aspect= aspect
        self.kWp_panel= 0.4 #kWp/panel
        self.area_panel= 1.7 #area/panel
        self.area_loss_factor = 0.5 #amout of area not exploitable
        self.footprint_area= footprint_area
        self.surface_area= self._surface_area()
        self.area_exploitable = self.surface_area * self.area_loss_factor
        self.loss= loss
        self.kwp= self._kwp()
        self.main_url= 'https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?'
        self.payload= {'lat': lat, 'lon': lon, 'peakpower': self.kwp,
                        'angle': self.angle, 'aspect': self.aspect, 'loss': self.loss,
                        'mountingplace': self.mountingplace,'outputformat':'json',
                       }
        self.r = requests.get(url=self.main_url, params=self.payload)
        self.pvgisdata = self.r.json()

    def _pvgisdata(self):
        print(self.main_url)
        r = requests.get(url=self.main_url, params=self.payload)
        return r.json()
    
    def _surface_area(self):
        angle_r = math.radians(self.angle)
        b = math.sqrt(self.footprint_area)
        hypotenus = b/math.cos(angle_r)
        surface_area = hypotenus * b
        return surface_area

    def _kwp(self):
        """
        se https://www.av-solkalkulator.no/calc
        :return: float, kilowattpeak
        """
        return 1

    def E_y(self):
        """
        Yearly PV energy production [kWh]
        :return: float, kilowatthours per square meter
        """
        # per kilowatt peak
        kWh_m2 = self.pvgisdata()['outputs']['totals']['fixed']['E_y']
        return kWh_m2

    def E_y_on_surface(self):
        """
        Yearly energy production kWh for exploitable surface area
        :return:
        """
        kWh_total = self.pvgisdata['outputs']['totals']['fixed']['E_y']*self.area_exploitable / self.area_panel * self.kWp_panel
        return kWh_total

    def Hi_y(self):
        """
        Average annual sum of global irradiation per square meter
        recieved by modules of the given system
        :return: float, kWh/m2/y
        """
        return self.pvgisdata['outputs']['totals']['fixed']['H(i)_y']

    def Hi_y_on_surface(self):
        """
        H(i)_y average per year for roof surface
        :return:
        """
        return self.pvgisdata['outputs']['totals']['fixed']['H(i)_y']


class Roof_hourly(Roof):
    """
    get hourly solar energy values from pv-gis for a given year.
    """
    def __init__(self, lat, lon, angle, aspect, footprint_area, loss = 14, mountingplace='free', pvcalc=True, startyear= 2019, endyear=2019):
        super().__init__(lat, lon, angle, aspect, footprint_area, loss, mountingplace)
        if pvcalc:
            pvcalc=1
        else:
            pvcalc=0
        self.main_url = 'https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?'
        self.startyear= startyear
        self.endyear = endyear
        self.payload = {'lat': lat, 'lon': lon, 'startyear': self.startyear, 'endyear': self.endyear,'pvcalculation': pvcalc,
                        'peakpower': self.kwp,'angle': self.angle, 'aspect': self.aspect, 'loss': self.loss,
                        'mountingplace': self.mountingplace, 'outputformat': 'json'}
        self.r = requests.get(self.main_url, params=self.payload)  
        self.pvgisdata = self.r.json()
        hourly = self.pvgisdata['outputs']['hourly']
        self.hourly_df = pd.DataFrame(hourly)

    def get_hourly_as_dataframe(self):
        return self.hourly_df

    def get_hourly_pivot(self):
        hourlydf = copy.deepcopy(self.hourly_df)
        hourlydf['datetime']= pd.to_datetime(hourlydf['time'], format= '%Y%m%d:%H%M')
        hourlydf['year'] = hourlydf['datetime'].dt.year
        piv= pd.pivot(data= hourlydf, columns= 'year', values= 'P')
        return piv

    def normalize(self, E_y_on_surface):
        normalized= copy.deepcopy(self.hourly_df)
        sum_year= normalized['P'].sum()
        normalized['normal']= normalized.apply(lambda x: (x.P /sum_year), axis= 'columns')
        normalized['p_normal']= normalized.apply(lambda x: x.normal*E_y_on_surface, axis= 'columns')
        return normalized

    def get_metadata(self):
        return self.pvgisdata['meta']
    
# -- Andre hjelpefunksjoner
def get_secret(filename):
    with open(filename) as file:
        secret = file.readline()
    return secret

def avrunding(tall):
    return int(round(tall, 2))

def dekningsberegning(DEKNINGSGRAD, timeserie, over_under = "over"):
    cutoff_effekt = max(timeserie)
    timeserie_sum = sum(timeserie)
    beregnet_dekningsgrad = 100.5
    while (beregnet_dekningsgrad / DEKNINGSGRAD) > 1:
        tmp_liste_h = np.zeros(8760)
        for i, timeverdi in enumerate(timeserie):
            if timeverdi > cutoff_effekt:
                tmp_liste_h[i] = cutoff_effekt
            else:
                tmp_liste_h[i] = timeverdi

        beregnet_dekningsgrad = (sum(tmp_liste_h) / timeserie_sum) * 100
        cutoff_effekt -= 0.05
    if over_under == "over":
        return tmp_liste_h
    if over_under == "under":
        return timeserie - tmp_liste_h

# -- Plotting
def plot_1_timeserie(
    timeserie,
    timeserie_navn,
    objektid,
    filplassering,
    COLOR="#1d3c34",
    VARIGHETSKURVE=False,
):
    if VARIGHETSKURVE == True:
        y_arr = np.sort(timeserie)[::-1]
        type_plot = "varighetskurve"
    else:
        y_arr = timeserie
        type_plot = "timeplot"
    x_arr = np.array(range(0, len(timeserie)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=y_arr,
            stackgroup="one",
            fill="tonexty",
            line=dict(width=0, color=COLOR),
        )
    )
    fig["data"][0]["showlegend"] = True
    fig["data"][0][
        "name"
    ] = f"{timeserie_navn}: {int(round(np.sum(y_arr),0)):,} kWh | {int(round(np.max(y_arr),0)):,} kW".replace(
        ",", " "
    )
    fig.update_xaxes(range=[0, 8760])
    fig.update_yaxes(range=[0, max(y_arr) * 1.1])
    fig.update_layout(barmode="stack")
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig.update_layout(
        xaxis_title="Timer i ett år", yaxis_title="Timesmidlet effekt [kWh/h]"
    )


def plot_2_timeserie(
    timeserie_1,
    timeserie_1_navn,
    timeserie_2,
    timeserie_2_navn,
    objektid,
    filplassering,
    COLOR_1="#1d3c34",
    COLOR_2="#4d4b32",
    VARIGHETSKURVE=False,
):
    if VARIGHETSKURVE == True:
        y_arr_1 = np.sort(timeserie_1)[::-1]
        y_arr_2 = np.sort(timeserie_2)[::-1]
        type_plot = "varighetskurve"
    else:
        y_arr_1 = timeserie_1
        y_arr_2 = timeserie_2
        type_plot = "timeplot"
    x_arr = np.array(range(0, len(timeserie_1)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=y_arr_1,
            stackgroup="one",
            fill="tonexty",
            line=dict(width=0, color=COLOR_1),
            name=f"{timeserie_1_navn}: {int(round(np.sum(y_arr_1),0)):,} kWh | {int(round(np.max(y_arr_1),0)):,} kW".replace(
                ",", " "
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=y_arr_2,
            stackgroup="one",
            fill="tonexty",
            line=dict(width=0, color=COLOR_2),
            name=f"{timeserie_2_navn}: {int(round(np.sum(y_arr_2),0)):,} kWh | {int(round(np.max(y_arr_2),0)):,} kW".replace(
                ",", " "
            ),
        )
    )
    fig["data"][0]["showlegend"] = True
    fig.update_xaxes(range=[0, 8760])
    fig.update_yaxes(range=[0, (max(y_arr_1) + max(y_arr_2)) * 1.1])
    fig.update_layout(barmode="stack")
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig.update_layout(
        xaxis_title="Timer i ett år", yaxis_title="Timesmidlet effekt [kWh/h]"
    )


def plot_3_timeserie(
    timeserie_1,
    timeserie_1_navn,
    timeserie_2,
    timeserie_2_navn,
    timeserie_3,
    timeserie_3_navn,
    objektid,
    filplassering,
    COLOR_1="#1d3c34",
    COLOR_2="#4d4b32",
    COLOR_3="#4da452",
    VARIGHETSKURVE=False,
):
    if VARIGHETSKURVE == True:
        y_arr_1 = np.sort(timeserie_1)[::-1]
        y_arr_2 = np.sort(timeserie_2)[::-1]
        y_arr_3 = np.sort(timeserie_3)[::-1]
        type_plot = "varighetskurve"
    else:
        y_arr_1 = timeserie_1
        y_arr_2 = timeserie_2
        y_arr_3 = timeserie_3
        type_plot = "timeplot"
    x_arr = np.array(range(0, len(timeserie_1)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=y_arr_1,
            stackgroup="one",
            fill="tonexty",
            line=dict(width=0, color=COLOR_1),
            name=f"{timeserie_1_navn}: {int(round(np.sum(y_arr_1),0)):,} kWh | {int(round(np.max(y_arr_1),0)):,} kW".replace(
                ",", " "
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=y_arr_2,
            stackgroup="one",
            fill="tonexty",
            line=dict(width=0, color=COLOR_2),
            name=f"{timeserie_2_navn}: {int(round(np.sum(y_arr_2),0)):,} kWh | {int(round(np.max(y_arr_2),0)):,} kW".replace(
                ",", " "
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=y_arr_3,
            stackgroup="one",
            fill="tonexty",
            line=dict(width=0, color=COLOR_3),
            name=f"{timeserie_3_navn}: {int(round(np.sum(y_arr_3),0)):,} kWh | {int(round(np.max(y_arr_2),0)):,} kW".replace(
                ",", " "
            ),
        )
    )
    fig["data"][0]["showlegend"] = True
    fig.update_xaxes(range=[0, 8760])
    fig.update_yaxes(range=[0, (max(y_arr_1) + max(y_arr_2) + max(y_arr_3)) * 1.1])
    fig.update_layout(barmode="stack")
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig.update_layout(
        xaxis_title="Timer i ett år", yaxis_title="Timesmidlet effekt [kWh/h]"
    )


# ---------------------------------------------------------------------------------------------------------------

# -- Klasse som tar inn energibehov fra PROFet-CSV
class Energibehov:
    BYGNINGSTYPE = {
        "A": "Hou",
        "B": "Apt",
        "C": "Off",
        "D": "Shp",
        "E": "Htl",
        "F": "Kdg",
        "G": "Sch",
        "H": "Uni",
        "I": "CuS",
        "J": "Nsh",
        "K": "Hospital",
        "L": "Other",
    }
    BYGNINGSSTANDARD = {"X": "Reg", "Y": "Eff-E", "Z": "Vef"}

    def __init__(self, objektid, bygningstype, bygningsstandard, areal):
        self.MAKS_VARMEEFFEKT_FAKTOR = 1.0  # todo: finner vi på noe lurt her? kan vi gange makseffekten med en faktor for å få reell maksimal effekt basert på utetemperatur feks?
        self.objektid = objektid
        self.bygningstype = bygningstype
        self.bygningsstandard = bygningsstandard
        self.areal = areal

    def _beregn_energibehov(self):
        oauth = OAuth2Session(client=BackendApplicationClient(client_id="profet_2023"))
        predict = OAuth2Session(token=oauth.fetch_token(token_url="https://identity.byggforsk.no/connect/token", client_id="profet_2023", client_secret=get_secret("energianalyse_secret.txt")))
        valgt_standard = self.BYGNINGSSTANDARD[self.bygningsstandard]
        if valgt_standard == "Reg":
            regular_areal = self.areal
            efficient_areal = 0
            veryefficient_areal = 0
        if valgt_standard == "Eff-E":
            regular_areal = 0
            efficient_areal = self.areal
            veryefficient_areal = 0
        if valgt_standard == "Vef":
            regular_areal = 0
            efficient_areal = 0
            veryefficient_areal = self.areal
        request_data = {
            "StartDate": "2022-01-01",              # Initial date (influences weekday/weekend. N.B. In Shops and Culture_Sport, Saturday has a different profile than Sunday)
            "Areas": {                              # Spesification of areas for building categories and efficiency levels
                f"{self.BYGNINGSTYPE[self.bygningstype.upper()]}": {                            # building category office, add sections for multiple categories. Available categories are ['Hou', 'Apt', 'Off', 'Shp', 'Htl', 'Kdg', 'Sch', 'Uni', 'CuS', 'Nsh', 'Hos', 'Other']
                    "Reg": regular_areal,                  # Category regular. 'Regular' means average standard of buildings in the stock
                    "Eff-E": efficient_areal,                # Category Efficient Existing. 'Efficient' means at about TEK10 standard, representing an ambitious yet realistic target for energy efficient renovation
                    "Eff-N": 0,                 # Category Efficient New. 'Efficient' means at about TEK10 standard. Gives same results as Eff-E
                    "Vef": veryefficient_areal                    # Category Very Efficient.'Very efficient' means at about passive house standard
                },
                # "Other": {                          # Category other represents the composition of the total norwegian building stock
                #     "Reg": 1000000,
                #     "Eff-E": 1000000,
                #     "Eff-N": 500000,
                #     "Vef": 500000
                # }
            },
            "RetInd": False,                        # Boolean, if True, individual profiles for each category and efficiency level are returned
            "Country": "Norway",                   # Optional, possiblity to get automatic holiday flags from the python holiday library.
        #     "TimeSeries": {                         # Time series input. If not used. 1 year standard Oslo climate is applied. If time series is included, prediction will be same length as input. Minimum 24 timesteps (hours)
        #         "Tout": [1.1, 1.1, 1.1, 0.9, 1.2, 1.1, 1.1, 1.3,
        #                  1.5, 1.6, 1.7, 1.6, 2.0, 2.2, 2.0, 2.4,
        #                  2.3, 2.3, 2.3, 2.2, 1.7, 1.4, 1.1, 1.6]
        # #        'HolidayFlag':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        #     }
        }
        r = predict.post("https://flexibilitysuite.byggforsk.no/api/Profet", json=request_data)
        data = r.json()
        df = pd.DataFrame.from_dict(data)
        df.reset_index(drop=True, inplace=True)
        self.df = df[["Electric", "DHW", "SpaceHeating"]]
        self.df.columns = ["Elspesifikt behov", "Tappevannsbehov", "Romoppvarmingsbehov"]
        self.romoppvarming_arr = df["SpaceHeating"]
        self.tappevann_arr = df["DHW"]
        self.el_spesifikk_arr = df["Electric"]

    def _nokkeltall(self):
        el_spesifikt_aarlig = avrunding(
            np.sum(self.df["Elspesifikt behov"])
        )
        tappevann_aarlig = avrunding(np.sum(self.df["Tappevannsbehov"]))
        romoppvarming_aarlig = avrunding(
            np.sum(self.df["Romoppvarmingsbehov"])
        )
        el_spesifikt_makseffekt = avrunding(
            np.max(self.df["Elspesifikt behov"])
        )
        tappevann_makseffekt = (
            avrunding(np.max(self.df["Tappevannsbehov"]))
            * self.MAKS_VARMEEFFEKT_FAKTOR
        )
        romoppvarming_makseffekt = (
            avrunding(np.max(self.df["Romoppvarmingsbehov"]))
            * self.MAKS_VARMEEFFEKT_FAKTOR
        )
        termisk_aarlig = romoppvarming_aarlig + tappevann_aarlig
        termisk_makseffekt = romoppvarming_makseffekt + tappevann_makseffekt
        return (
            el_spesifikt_aarlig,
            tappevann_aarlig,
            romoppvarming_aarlig,
            termisk_aarlig,
            el_spesifikt_makseffekt,
            tappevann_makseffekt,
            romoppvarming_makseffekt,
            termisk_makseffekt,
        )

    def _visualisering(self):
        FILPLASSERING = r"C:/Users/magne.syljuasen/Progg/INTOZERO2/"
        EL_SPESIFIKK_FARGE = "#b7dc8f"
        ROMOPPVARMING_FARGE = "#1d3c34"
        TAPPEVANN_FARGE = "#FFC358"
        plot_1_timeserie(
            self.el_spesifikk_arr,
            "Elspesifikt behov",
            self.objektid,
            FILPLASSERING,
            COLOR=EL_SPESIFIKK_FARGE,
        )
        plot_1_timeserie(
            self.tappevann_arr,
            "Tappevannsbehov",
            self.objektid,
            FILPLASSERING,
            COLOR=TAPPEVANN_FARGE,
        )
        plot_1_timeserie(
            self.romoppvarming_arr,
            "Romoppvarmingsbehov",
            self.objektid,
            FILPLASSERING,
            COLOR=ROMOPPVARMING_FARGE,
        )
        plot_1_timeserie(
            self.el_spesifikk_arr,
            "Elspesifikt behov",
            self.objektid,
            FILPLASSERING,
            COLOR=TAPPEVANN_FARGE,
            VARIGHETSKURVE=True,
        )
        plot_1_timeserie(
            self.tappevann_arr,
            "Tappevannsbehov",
            self.objektid,
            FILPLASSERING,
            COLOR=TAPPEVANN_FARGE,
            VARIGHETSKURVE=True,
        )
        plot_1_timeserie(
            self.romoppvarming_arr,
            "Romoppvarmingsbehov",
            self.objektid,
            FILPLASSERING,
            COLOR=ROMOPPVARMING_FARGE,
            VARIGHETSKURVE=True,
        )
        # -- sammenstilte
        plot_2_timeserie(
            self.tappevann_arr,
            "Tappevannsbehov",
            self.romoppvarming_arr,
            "Romoppvarmingsbehov",
            self.objektid,
            FILPLASSERING,
            COLOR_1=TAPPEVANN_FARGE,
            COLOR_2=ROMOPPVARMING_FARGE,
        )
        plot_2_timeserie(
            self.tappevann_arr,
            "Tappevannsbehov",
            self.romoppvarming_arr,
            "Romoppvarmingsbehov",
            self.objektid,
            FILPLASSERING,
            COLOR_1=TAPPEVANN_FARGE,
            COLOR_2=ROMOPPVARMING_FARGE,
            VARIGHETSKURVE=True,
        )
        plot_3_timeserie(
            self.tappevann_arr,
            "Tappevannsbehov",
            self.el_spesifikk_arr,
            "Elspesifikt behov",
            self.romoppvarming_arr,
            "Romoppvarmingsbehov",
            self.objektid,
            FILPLASSERING,
            COLOR_1="#1d3c34",
            COLOR_2=EL_SPESIFIKK_FARGE,
            COLOR_3=ROMOPPVARMING_FARGE,
        )
        plot_3_timeserie(
            self.tappevann_arr,
            "Tappevannsbehov",
            self.el_spesifikk_arr,
            "Elspesifikt behov",
            self.romoppvarming_arr,
            "Romoppvarmingsbehov",
            self.objektid,
            FILPLASSERING,
            COLOR_1="#1d3c34",
            COLOR_2=EL_SPESIFIKK_FARGE,
            COLOR_3=ROMOPPVARMING_FARGE,
            VARIGHETSKURVE=True,
        )

    def _lagring(self, timeserier_obj):
        timeserier_obj.legg_inn_timeserie(timeserie=self.el_spesifikk_arr, timeserie_navn="El_spesifiktbehov")
        timeserier_obj.legg_inn_timeserie(timeserie=self.romoppvarming_arr, timeserie_navn="R_omoppvarmingsbehov")
        timeserier_obj.legg_inn_timeserie(timeserie=self.tappevann_arr, timeserie_navn="V_armtvannsbehov")

    def standard_metode(self, lagring_obj):
        self._beregn_energibehov()
        #self._visualisering()
        self._lagring(lagring_obj)
        return self._nokkeltall()

# Ulike tiltak
# --
class Fjernvarme:
    def __init__(
        self,
        objektid,
        DEKNINGSGRAD,
        behovstype,
        df,
        VIRKNINGSGRAD=100,
    ):
        self.objektid = objektid
        self.df = df
        self.DEKNINGSGRAD = DEKNINGSGRAD
        self.VIRKNINGSGRAD = VIRKNINGSGRAD / 100  # i prosent
        self.behovstype = behovstype
        if behovstype == "V":
            self.termisk_arr = dekningsberegning(DEKNINGSGRAD=self.DEKNINGSGRAD, timeserie=(df["Tappevannsbehov"]))
        if behovstype == "R":
            self.termisk_arr = dekningsberegning(DEKNINGSGRAD=self.DEKNINGSGRAD, timeserie=(df["Romoppvarmingsbehov"]))
        if behovstype == "T":
            self.termisk_arr = dekningsberegning(DEKNINGSGRAD=self.DEKNINGSGRAD, timeserie=(df["Romoppvarmingsbehov"] + df["Tappevannsbehov"]))

    def _beregn_fjernvarme(self):
        self.fjernvarme_arr = (self.termisk_arr) * self.VIRKNINGSGRAD
        self.df["Fjernvarme"] = self.fjernvarme_arr

    def _visualisering(self):
        FILPLASSERING = r"C:/Users/magne.syljuasen/Progg/INTOZERO2/"
        FJERNVARME_FARGE = "#00FFFF"
        plot_1_timeserie(
            self.fjernvarme_arr,
            "Fjernvarmedekning",
            self.objektid,
            FILPLASSERING,
            COLOR=FJERNVARME_FARGE,
        )
        plot_1_timeserie(
            self.fjernvarme_arr,
            "Fjernvarmedekning",
            self.objektid,
            FILPLASSERING,
            COLOR=FJERNVARME_FARGE,
            VARIGHETSKURVE=True,
        )

    def _nokkeltall(self):
        self.fjernvarme_aarlig = avrunding(np.sum(self.fjernvarme_arr))
        self.fjernvarme_makseffekt = avrunding(np.max(self.fjernvarme_arr))
        return self.fjernvarme_aarlig, self.fjernvarme_makseffekt

    def _lagring(self, timeserie_obj):
        timeserie_obj.legg_inn_timeserie(timeserie=-self.fjernvarme_arr,timeserie_navn=f"{self.behovstype}_fjernvarme")

    def standard_metode(self, lagring_obj):
        self._beregn_fjernvarme()
        #self._visualisering()
        self._lagring(timeserie_obj=lagring_obj)
        return self._nokkeltall()

# --
class Grunnvarme:
    def __init__(self,
        objektid,
        behovstype,
        df,
        COP,
        DEKNINGSGRAD
    ):
        self.objektid = objektid
        self.DEKNINGSRAD = DEKNINGSGRAD
        self.COP = COP
        self.behovstype = behovstype
        if behovstype == "V":
            self.termisk_arr = dekningsberegning(DEKNINGSGRAD=self.DEKNINGSRAD, timeserie=(df["Tappevannsbehov"]))
        if behovstype == "R":
            self.termisk_arr = dekningsberegning(DEKNINGSGRAD=self.DEKNINGSRAD, timeserie=(df["Romoppvarmingsbehov"]))
        if behovstype == "T":
            self.termisk_arr = dekningsberegning(DEKNINGSGRAD=self.DEKNINGSRAD, timeserie=(df["Romoppvarmingsbehov"] + df["Tappevannsbehov"]))

    def _beregn_grunnvarme(self):
        self.varmepumpe_arr = dekningsberegning(DEKNINGSGRAD=self.DEKNINGSRAD, timeserie=(self.termisk_arr))
        self.levert_fra_bronner_arr = self.varmepumpe_arr - self.varmepumpe_arr/self.COP #todo : timevariert COP, da må vi også hente inn informasjon om utetemperatur
        self.kompressor_arr = self.varmepumpe_arr - self.levert_fra_bronner_arr
        self.spisslast_arr = self.termisk_arr - self.varmepumpe_arr
        
    def _visualisering(self):
        FILPLASSERING = r"C:/Users/magne.syljuasen/Progg/INTOZERO2/"
        KOMPRESSOR_FARGE = "#1d3c34"
        LEVERT_FRA_BRONNER_FARGE = "#b7dc8f"
        SPISSLAST_FARGE = "#FFC358"
        plot_3_timeserie(timeserie_1=self.kompressor_arr, timeserie_1_navn="Strøm til varmepumpe", timeserie_2=self.levert_fra_bronner_arr, timeserie_2_navn="Levert fra brønner", timeserie_3=self.spisslast_arr, timeserie_3_navn="Spisslast", objektid=self.objektid, filplassering=FILPLASSERING, COLOR_1=KOMPRESSOR_FARGE, COLOR_2=LEVERT_FRA_BRONNER_FARGE, COLOR_3=SPISSLAST_FARGE)
        plot_3_timeserie(timeserie_1=self.kompressor_arr, timeserie_1_navn="Strøm til varmepumpe", timeserie_2=self.levert_fra_bronner_arr, timeserie_2_navn="Levert fra brønner", timeserie_3=self.spisslast_arr, timeserie_3_navn="Spisslast", objektid=self.objektid, filplassering=FILPLASSERING, COLOR_1=KOMPRESSOR_FARGE, COLOR_2=LEVERT_FRA_BRONNER_FARGE, COLOR_3=SPISSLAST_FARGE, VARIGHETSKURVE=True)

    def _nokkeltall(self):
        kompressor_aarlig = avrunding(np.sum(self.kompressor_arr))
        levert_fra_bronner_aarlig = avrunding(np.sum(self.levert_fra_bronner_arr))
        spisslast_aarlig = avrunding(np.sum(self.spisslast_arr))
        kompressor_makseffekt = avrunding(np.max(self.kompressor_arr))
        levert_fra_bronner_makseffekt = avrunding(np.max(self.levert_fra_bronner_arr))
        spisslast_makseffekt = avrunding(np.max(self.spisslast_arr))
        return kompressor_aarlig, levert_fra_bronner_aarlig, spisslast_aarlig, kompressor_makseffekt, levert_fra_bronner_makseffekt, spisslast_makseffekt
    
    def _lagring(self, timeserie_obj):
        timeserie_obj.legg_inn_timeserie(timeserie=-(self.levert_fra_bronner_arr + self.kompressor_arr),timeserie_navn=f"{self.behovstype}_grunnvarme")
        timeserie_obj.legg_inn_timeserie(timeserie=self.kompressor_arr,timeserie_navn=f"El_grunnvarme_kompressor")
        timeserie_obj.legg_inn_timeserie(timeserie=self.spisslast_arr,timeserie_navn=f"El_kjel")

    def standard_metode(self, lagring_obj):
        self._beregn_grunnvarme()
        #self._visualisering()
        self._lagring(timeserie_obj=lagring_obj)
        return self._nokkeltall()

# --
class LuftLuftVarmepumpe:
    def __init__(self,
        objektid,
        df,
        COP=2.8,
        DEKNINGSGRAD=80
    ):
        self.objektid = objektid
        self.DEKNINGSRAD = DEKNINGSGRAD
        self.COP = COP
        self.termisk_arr = df["Romoppvarmingsbehov"]

    def _beregn_luft_luft_varmepumpe(self):
        self.varmepumpe_arr = dekningsberegning(DEKNINGSGRAD=self.DEKNINGSRAD, timeserie=(self.termisk_arr))
        self.levert_fra_luft_arr = self.varmepumpe_arr - self.varmepumpe_arr/self.COP #todo : timevariert COP, da må vi også hente inn informasjon om utetemperatur
        self.kompressor_arr = self.varmepumpe_arr - self.levert_fra_luft_arr
        self.spisslast_arr = self.termisk_arr - self.varmepumpe_arr
        
    def _visualisering(self):
        FILPLASSERING = r"C:/Users/magne.syljuasen/Progg/INTOZERO2/"
        KOMPRESSOR_FARGE = "#1d3c34"
        LEVERT_FRA_LUFT_FARGE = "#b7dc8f"
        SPISSLAST_FARGE = "#FFC358"
        plot_3_timeserie(timeserie_1=self.kompressor_arr, timeserie_1_navn="Strøm til varmepumpe", timeserie_2=self.levert_fra_luft_arr, timeserie_2_navn="Levert fra luft", timeserie_3=self.spisslast_arr, timeserie_3_navn="Spisslast", objektid=self.objektid, filplassering=FILPLASSERING, COLOR_1=KOMPRESSOR_FARGE, COLOR_2=LEVERT_FRA_LUFT_FARGE, COLOR_3=SPISSLAST_FARGE)
        plot_3_timeserie(timeserie_1=self.kompressor_arr, timeserie_1_navn="Strøm til varmepumpe", timeserie_2=self.levert_fra_luft_arr, timeserie_2_navn="Levert fra luft", timeserie_3=self.spisslast_arr, timeserie_3_navn="Spisslast", objektid=self.objektid, filplassering=FILPLASSERING, COLOR_1=KOMPRESSOR_FARGE, COLOR_2=LEVERT_FRA_LUFT_FARGE, COLOR_3=SPISSLAST_FARGE, VARIGHETSKURVE=True)

    def _nokkeltall(self):
        kompressor_aarlig = avrunding(np.sum(self.kompressor_arr))
        levert_fra_luft_aarlig = avrunding(np.sum(self.levert_fra_luft_arr))
        spisslast_aarlig = avrunding(np.sum(self.spisslast_arr))
        
        kompressor_makseffekt = avrunding(np.max(self.kompressor_arr))
        levert_fra_luft_makseffekt = avrunding(np.max(self.levert_fra_luft_arr))
        spisslast_makseffekt = avrunding(np.max(self.spisslast_arr))

        return kompressor_aarlig, levert_fra_luft_aarlig, spisslast_aarlig, kompressor_makseffekt, levert_fra_luft_makseffekt, spisslast_makseffekt
    
    def _lagring(self, timeserie_obj):
        timeserie_obj.legg_inn_timeserie(timeserie=-self.levert_fra_luft_arr,timeserie_navn=f"R_luftluft")
        timeserie_obj.legg_inn_timeserie(timeserie=self.kompressor_arr,timeserie_navn=f"El_luftluft_kompressor")
        timeserie_obj.legg_inn_timeserie(timeserie=self.spisslast_arr,timeserie_navn=f"El_luftluft_spisslast")

    def standard_metode(self, lagring_obj):
        self._beregn_luft_luft_varmepumpe()
        #self._visualisering()
        self._lagring(timeserie_obj=lagring_obj)
        return self._nokkeltall()
        

# --
class Solproduksjon:
    def __init__(self, objektid: int, lat: float, lon: float, takflate_navn: list, takflate_vinkel: int, takflate_arealer: list,
                 takflate_orienteringer: list, loss = 14, mountingplace= 'free', angle = None, startyear = 2019, endyear = 2019):
        """
        :param objektid: int - objektid for bygget
        :param lat: latidtude
        :param lon: longitude
        :param takflate_navn: list - liste med navn på takflate  [A,B,C,D]. Standardisert i byggtabell
        :param takflate_vinkel: list - liste med vinkler tilhørende takflate_navn
        :param takflate_arealer: list - liste med arealer tilhørende takflate_navn
        :param takflate_orienteringer: list - liste med orientering tilhørende takflate_navn
        :param loss: int - tap (pvgis)
        :param mountingplace: str - free, building
        """
        self.objektid = objektid
        self.lat = lat
        self.lon = lon
        self.takflate_vinkel = takflate_vinkel
        self.takflate_navn = takflate_navn
        self.takflate_arealer = takflate_arealer
        self.takflate_orienteringer = takflate_orienteringer
        self._validate_lists_equal_lengt()
        self.loss= loss
        self.startyear = startyear
        self.endyear = endyear
        if not angle:
            self.angle = 25
        else:
            self.angle= angle
        self.mountingplace = mountingplace
        self.takflate_pv_objs = {}
        self._timeserier_dataframes = []

    def _calculate_pv_on_one_roof_part(self, takflate_navn, aspect, footprint_area):
        # returnere timeserier
        if aspect and footprint_area:
            roof= Roof(lat= self.lat,
                       lon= self.lon,
                       angle= self.angle,
                       aspect= aspect,
                       footprint_area= footprint_area,
                       loss= self.loss,
                       mountingplace= self.mountingplace)
            roof_hourly = Roof_hourly(lat=self.lat, lon=self.lon, angle=self.angle,
                                            aspect=aspect, footprint_area=footprint_area,
                                            loss=self.loss, mountingplace= self.mountingplace,
                                            pvcalc= True, startyear= self.startyear, endyear= self.endyear)
            normalized = roof_hourly.normalize(E_y_on_surface=roof.E_y_on_surface())
            self._timeserier_dataframes.append(normalized[['P', 'normal', 'p_normal']])
            self.takflate_pv_objs[takflate_navn]= roof

    def _validate_lists_equal_lengt(self):
        takflater = len(self.takflate_navn)
        orienteringer = len(self.takflate_orienteringer)
        arealer = len(self.takflate_arealer)
        if orienteringer != takflater:
            raise RoofValueListOrienteringNotEqualLengthError(orienteringer)
        if arealer != takflater:
            raise RoofValueListArealerNotEqualLengthError(arealer)
    
    def _calculate_whole_roof(self):
        for takflatenavn, aspect, footprint_area in zip(self.takflate_navn, self.takflate_orienteringer, self.takflate_arealer):
            self._calculate_pv_on_one_roof_part(takflate_navn= takflatenavn, aspect= aspect, footprint_area= footprint_area)
            
    def _timesserie(self):
        normalized_hourly_sum= sum(self._timeserier_dataframes)
        return normalized_hourly_sum.p_normal

    def _visualisering(self):
        FILPLASSERING = r"C:/Users/magne.syljuasen/Progg/INTOZERO2/"
        SOL_FARGE = "#b7dc8f"
        plot_1_timeserie(
            self._timesserie(),
            "Solproduksjon",
            self.objektid,
            FILPLASSERING,
            COLOR=SOL_FARGE,
        )

    def _nokkeltall(self):
        e_y_sum= sum([takflate_pv_objs.E_y_on_surface() for takflate_pv_objs in self.takflate_pv_objs.values()])
        return self.takflate_pv_objs, e_y_sum
    
    def _lagring(self, timeserie_obj):
        timeserie_obj.legg_inn_timeserie(timeserie=-self._timesserie(), timeserie_navn=f"El_solenergi")
    
    def standard_metode(self, lagring_obj):
        self._calculate_whole_roof()
        #self._visualisering()
        self._lagring(timeserie_obj=lagring_obj)
        return self._nokkeltall()
        

# hovedmodul som tar inn energimix
# tallet som input kan varierere mellom 0 og 100 (%)
# ---
class Energianalyse:
    def __init__(
        self,
        objektid,
        energibehov_start_beregning : bool,
        energibehov_bygningstype : str,
        energibehov_bygningsstandard : str,
        energibehov_areal : int,
        grunnvarme_start_beregning : bool,
        grunnvarme_energibehov : str, #gyldige input er "T": termisk, "V": varmtvann og "R": romoppvarming
        grunnvarme_dekningsgrad : int, #tall fra 0 - 100
        grunnvarme_cop : float, #årsvarmefaktor,
        fjernvarme_start_beregning : bool, 
        fjernvarme_energibehov : int, #gyldige input er "T": termisk, "V": varmtvann og "R": romoppvarming 
        fjernvarme_dekningsgrad : int, #tall fra 0 - 100
        luft_luft_start_beregning : bool,
        luft_luft_cop : float, #årsvarmefaktor
        luft_luft_dekningsgrad : int, #tall fra 0 - 100
        solproduksjon_start_beregning : bool,
        solproduksjon_lat : int, #latitude
        solproduksjon_lon : int, #longitude
        solproduksjon_takflate_vinkel : int, # feks. 30
        solproduksjon_takflate_navn : list, # feks. ["A", "B"]
        solproduksjon_takflate_arealer : list, #feks. [10, 10]
        solproduksjon_takflate_orienteringer : list #feks. [90, -90]
    ):
        self.objektid = objektid
        self.timeserier_obj = Timeserier()
        if energibehov_start_beregning == True:
            self.energibehov_obj = Energibehov(
                objektid=objektid,
                bygningstype=energibehov_bygningstype,
                bygningsstandard=energibehov_bygningsstandard,
                areal=energibehov_areal,
            )
            self.energibehov_obj.standard_metode(lagring_obj=self.timeserier_obj)
        if fjernvarme_start_beregning == True:
            self.fjernvarme_obj = Fjernvarme(
                objektid=objektid,
                DEKNINGSGRAD=fjernvarme_dekningsgrad,
                behovstype=fjernvarme_energibehov,
                df = self.energibehov_obj.df,
                VIRKNINGSGRAD=100
            )
            self.fjernvarme_obj.standard_metode(lagring_obj=self.timeserier_obj)
        if grunnvarme_start_beregning == True:
            self.grunnvarme_obj = Grunnvarme(objektid=objektid, behovstype=grunnvarme_energibehov, df=self.energibehov_obj.df, COP = grunnvarme_cop, DEKNINGSGRAD = grunnvarme_dekningsgrad)
            self.grunnvarme_obj.standard_metode(lagring_obj=self.timeserier_obj)
        if luft_luft_start_beregning == True:
            self.luft_luft_obj = LuftLuftVarmepumpe(objektid=objektid, df=self.energibehov_obj.df, COP=luft_luft_cop, DEKNINGSGRAD=luft_luft_dekningsgrad)
            self.luft_luft_obj.standard_metode(lagring_obj=self.timeserier_obj)
        if solproduksjon_start_beregning == True:
            self.solproduksjon_obj = Solproduksjon(objektid=objektid, lat = solproduksjon_lat, lon = solproduksjon_lon, takflate_navn = solproduksjon_takflate_navn, takflate_vinkel = solproduksjon_takflate_vinkel, takflate_arealer=solproduksjon_takflate_arealer, takflate_orienteringer=solproduksjon_takflate_orienteringer)
            self.solproduksjon_obj.standard_metode(lagring_obj=self.timeserier_obj)
        #--
        self._sammenstilling()
        #self._visualisering()
             
    def _sammenstilling(self):
        df = self.timeserier_obj.df
        # Group the columns based on the string split value ('_') of the column names
        grouped_cols = df.columns.str.split('_', expand=True)
        grouped_cols = [x[0] for x in grouped_cols.to_numpy()]
        unique_groups = sorted(list(set(grouped_cols)))

        # Add all arrays in the DataFrame with similar string split name
        new_df = pd.DataFrame()
        for group in unique_groups:
            #print(group)
            cols = [col for col in df.columns if col.startswith(group)]
            new_df[group] = df[cols].sum(axis=1)

        if "T" in new_df.columns:
            new_df["T"] = new_df["V"] + new_df["R"] + new_df["T"]
            new_df.drop(['V', 'R'], inplace=True, axis=1)
        self.new_df = new_df
        #--
        positive_rows = self.timeserier_obj.df[self.timeserier_obj.df>0]
        self.df_before = positive_rows.sum(axis=1)
        self.df_after = self.new_df.sum(axis=1)

    def _visualisering(self):
        FILPLASSERING = r"C:/Users/magne.syljuasen/Progg/INTOZERO2/"
        KOMPRESSOR_FARGE = "#1d3c34"
        LEVERT_FRA_LUFT_FARGE = "#b7dc8f"
        SPISSLAST_FARGE = "#FFC358"

        plot_1_timeserie(
            self.new_df["El"],
            "a",
            self.objektid,
            FILPLASSERING,
            COLOR=SPISSLAST_FARGE,
        )
    def _nokkeltall(self):
        pass

    def _lagring(self, objektid):
        # skriver de ut de relevante timeseriene til tabell
        pass


# tester
# flere bygg
ANTALL_BYGG = 3
st.title("Demo - Beregningsmodul Into Zero")
#lists
objektid_list = [1, 2, 3]
c1, c2 = st.columns(2)
energibehov_start_beregning_list = True
energibehov_bygningstype_list = "A"
energibehov_areal_list = st.number_input("Bygningsareal [m2]", value = 250)
energibehov_bygningsstandard_list = st.selectbox("Velg bygningsstandard (X = eldre bygg, Y = TEK10, Z = passivhus)", options=["X", "Y", "Z"])
grunnvarme_energibehov_list = "T"
grunnvarme_dekningsgrad_list = 90
gv = st.checkbox("Grunnvarme")
if gv:
    grunnvarme_start_beregning_list = True
    grunnvarme_cop_list = st.number_input("Velg COP", min_value=2.0, value=3.5, max_value=5.0)
else:
    grunnvarme_start_beregning_list = False
    grunnvarme_cop_list = 3
fjernvarme_energibehov_list = "T"
fjernvarme_dekningsgrad_list = 100
f = st.checkbox("Fjernvarme")
if f:
    fjernvarme_start_beregning_list = True
else:
    fjernvarme_start_beregning_list = False
if f and gv:
    st.error("Velg kun en varmeforsyningsløsning")
    st.stop()
luft_luft_start_beregning_list = False
luft_luft_cop_list = 2.8
luft_luft_dekningsgrad_list = 80
#--
solproduksjon_lat_list = 62
solproduksjon_lon_list = 10
solproduksjon_takflate_navn_list = [["A", "B"]]
solproduksjon_takflate_orienteringer_list = [[90,-90]]
if st.checkbox("Solceller"):
    solproduksjon_start_beregning_list = True
    solproduksjon_takflate_vinkel = st.number_input("Velg takvinkel [grader]", value = 45)
    solproduksjon_takflate_arealer_list = st.number_input("Velg takareal [m2]", value = 50)
    sol_areal_1 = solproduksjon_takflate_arealer_list/2
    sol_areal_2 = solproduksjon_takflate_arealer_list/2
else:
    solproduksjon_start_beregning_list = False
    solproduksjon_takflate_vinkel = 0
    solproduksjon_takflate_arealer_list = [10, 10]
    sol_areal_1 = 0
    sol_areal_2 = 0
if st.button("Start beregning"):
    st.markdown("---")
    with st.spinner("Beregner..."):
        #--
        start_time = time.time()
        #st.caption(f"Grunnvarme : {grunnvarme_start_beregning_list}")
        #st.caption(f"Fjernvarme : {fjernvarme_start_beregning_list}")
        #st.caption(f"Luft-luft-varmepumpe : {luft_luft_start_beregning_list}")
        #st.caption(f"Solceller : {solproduksjon_start_beregning_list}")
        energi_obj = Energianalyse(
            objektid=1,
            energibehov_start_beregning = energibehov_start_beregning_list,
            energibehov_bygningstype = energibehov_bygningstype_list,
            energibehov_bygningsstandard = energibehov_bygningsstandard_list ,
            energibehov_areal = energibehov_areal_list ,
            grunnvarme_start_beregning = grunnvarme_start_beregning_list ,
            grunnvarme_energibehov = grunnvarme_energibehov_list ,
            grunnvarme_cop = grunnvarme_cop_list ,
            grunnvarme_dekningsgrad = grunnvarme_dekningsgrad_list ,
            fjernvarme_start_beregning = fjernvarme_start_beregning_list ,
            fjernvarme_energibehov = fjernvarme_energibehov_list ,
            fjernvarme_dekningsgrad = fjernvarme_dekningsgrad_list ,
            luft_luft_start_beregning = luft_luft_start_beregning_list ,
            luft_luft_cop = luft_luft_cop_list ,
            luft_luft_dekningsgrad = luft_luft_dekningsgrad_list ,
            solproduksjon_start_beregning = solproduksjon_start_beregning_list ,
            solproduksjon_lat = solproduksjon_lat_list , 
            solproduksjon_lon = solproduksjon_lon_list , 
            solproduksjon_takflate_vinkel = solproduksjon_takflate_vinkel , 
            solproduksjon_takflate_navn = ["A", "B"] , 
            solproduksjon_takflate_arealer = [sol_areal_1, sol_areal_2] , 
            solproduksjon_takflate_orienteringer = [90, -90]     
        )
        #st.write(f"**Før tiltak: {int(np.sum(energi_obj.timeserier_obj.df))} kWh**")
        st.caption(f"El-spesifikt behov: {int(np.sum(energi_obj.timeserier_obj.df['El_spesifiktbehov']))} kWh/år")
        st.caption(f"Romoppvarmingsbehov: {int(np.sum(energi_obj.timeserier_obj.df['R_omoppvarmingsbehov']))} kWh/år")
        st.caption(f"Varmtvannsbehov: {int(np.sum(energi_obj.timeserier_obj.df['V_armtvannsbehov']))} kWh/år")
        st.write(f"**Utgangspunkt: Alle energibehovene dekkes av strøm {int(np.sum(energi_obj.timeserier_obj.df['El_spesifiktbehov'])) + int(np.sum(energi_obj.timeserier_obj.df['R_omoppvarmingsbehov'])) + int(np.sum(energi_obj.timeserier_obj.df['V_armtvannsbehov']))} kWh/år**")
        st.markdown("---")
        st.write(f"**Timeserier inkludert tiltak (grunnvarme, fjernvarme, solceller)**")
        with chart_container(energi_obj.timeserier_obj.df):
            st.bar_chart(energi_obj.timeserier_obj.df)
        #st.bar_chart(energi_obj.df_before)
        st.markdown("---")
        st.write(f"**Timeserier lagt sammen (gjenstående strøm): {int(np.sum(energi_obj.df_after))} kWh**")
        with chart_container(energi_obj.new_df):
            st.bar_chart(energi_obj.new_df)
        #st.bar_chart(energi_obj.df_after)

        end_time = time.time()
        elapsed_time = end_time - start_time

        #st.write(energi_obj.fjernvarme_obj.fjernvarme_aarlig)