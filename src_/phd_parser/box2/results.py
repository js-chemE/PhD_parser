import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


CARBON_SPECIES_INFO = {
    "H2": 0,
    "CO2": 1,
    "CO": 1,
    "MeOH": 1,
    "MF": 2,
    "DME": 2,
    "CH4": 1,
    "EtOH": 2,
    "Ar": 0,
    "N2": 0
    }

HYDROGEN_SPECIES_INFO = {
    "H2": 2,
    "CO2": 0,
    "CO": 0,
    "MeOH": 4,
    "MF": 6,
    "DME": 6,
    "CH4": 4,
    "EtOH": 6,
    "Ar": 0,
    "N2": 0
    }

SPECIES = ["Ar", "N2", "CO2", "H2", "MeOH", "MF", "CO", "DME", "CH4", "EtOH"]
SPECIES_INERT = ["Ar", "N2"]
SPECIES_REACTANTS = ["CO2", "H2"]
SPECIES_PRODUCTS = ["MeOH", "MF", "CO", "DME", "CH4", "EtOH"]

class Box2Results(BaseModel):
    labview: pd.DataFrame | None = Field(default=None)
    gc_raw: pd.DataFrame | None = Field(default=None)
    gc: pd.DataFrame | None = Field(default=None)
    grouped_mean: pd.DataFrame | None = Field(default=None)
    grouped_std: pd.DataFrame | None = Field(default=None)
    species: List[str] | None = Field(default=SPECIES)
    species_inert: List[str] | None = Field(default=SPECIES_INERT)
    species_reactants: List[str] | None = Field(default=SPECIES_REACTANTS)
    species_products: List[str] | None = Field(default=SPECIES_PRODUCTS)

    feed_info: Dict[str, float] | None = Field(default=None)
    intervals: List[Tuple[float, float]] | List[List[float]] | None = Field(default=None)
    carbon_species_info: Dict[str, int] | None = Field(default=CARBON_SPECIES_INFO)
    hydrogen_species_info: Dict[str, int] | None = Field(default=HYDROGEN_SPECIES_INFO)
    raw_data: pd.DataFrame | None = Field(default=None)
    data: pd.DataFrame | None = Field(default=None)

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def feed_species(self, species: str | List[str]) -> float | np.ndarray | None:
        if isinstance(species, str):
            try:
                return self.feed_info[species]
            except KeyError:
                logger.warning(f"Species {species} not found in feed_info")
                return 0
        elif isinstance(species, list):
            try:
                return np.ndarray([self.feed_info[specie] for specie in species])
            except KeyError:
                logger.warning(f"Species {species} not found in feed_info")
                return 0

    def process_gc(
            self,
            species_standard: str = "N2"):
        
        self.gc = self.gc_raw.copy()
        self.gc.drop(
            columns=["Method", "Sample Id", "File Name"],
            errors="ignore",
            inplace=True
        )
        self.gc["sum"] = 0
        for species in self.species:
            self.gc["sum"] += self.gc[species]
        
        try:
            self.gc["alpha"] = self.gc[species_standard] / self.feed_species(species_standard)
            if self.gc["alpha"].isnull().any() or (self.gc["alpha"] == 0).any():
                logger.error("Alpha calculation resulted in NaN or zero. Check feed values.")
        except Exception as e:
            logger.error(f"Error in alpha calculation: {e}")
        

        """Molar Balance"""
        self.gc["molar_in"] = 0
        for species in self.species:
            self.gc["molar_in"] += self.feed_species(species)
        self.gc["molar_out"] = 0
        for species in self.species:
            self.gc["molar_out"] += self.gc[species] / self.gc["alpha"]
        self.gc["molar_balance"] = self.gc["molar_out"]/ self.gc["molar_in"]

        """Carbon Balance"""
        self.gc["carbon_in"] = 0
        for species in self.species:
            self.gc["carbon_in"] += self.carbon_species_info.get(species, 0) * self.feed_info.get(species, 0)

        self.gc["carbon_out"] = 0
        for species in self.species:
            self.gc["carbon_out"] += self.gc[species] / self.gc["alpha"] * self.carbon_species_info.get(species, 0)
        
        
        self.gc["carbon_balance"] = self.gc["carbon_out"] / self.gc["carbon_in"] 

        logger.info(f"Carbon Inlet: {self.gc['carbon_in'].mean():.2f} | Carbon Outlet: {self.gc['carbon_out'].mean():.2f}")
        logger.info(f"Carbon Balance Mean: {self.gc['carbon_balance'].mean():.2f}")

        """Conversion"""
        if self.species_reactants is not None:
            for reactant in self.species_reactants:
                self.gc["X_" + reactant] = 1 - (1 / self.gc["alpha"] * (self.gc[reactant] / self.feed_species(reactant)))
        
            if "CO2" in self.species_reactants:
                sum_products = np.zeros_like(self.gc["CO2"].values)
                for product in self.species_products:
                    sum_products += self.gc[product] * self.carbon_species_info.get(product, 0)
                self.gc["XP_CO2"] = 1 / self.gc["alpha"] * (sum_products / (self.feed_species("CO2") * self.carbon_species_info.get("CO2", 0)))
            
            if "H2" in self.species_reactants:
                sum_products = np.zeros_like(self.gc["H2"].values)
                for product in self.species_products:
                    sum_products += self.gc[product] * self.hydrogen_species_info.get(product, 0)
                self.gc["XP_H2"] = 1 / self.gc["alpha"] * (sum_products / (self.feed_species("H2") * self.hydrogen_species_info.get("H2", 0)))

        """Selectivity"""
        self.gc["carbon_sum_products"] = 0
        for product in self.species_products:
            self.gc["carbon_sum_products"] += self.gc[product] * self.carbon_species_info.get(product, 0)

        # Carbon Selectivity
        for product in self.species_products:
            self.gc["S_" + product] = self.gc[product] * self.carbon_species_info.get(product, 0) / self.gc["carbon_sum_products"]
        
        # Carbon Selectivity based on CO2 consumed
        for reactant in self.species_reactants:
            self.gc[reactant + "_delta"] = (self.feed_species(reactant) - (self.gc[reactant] / self.gc["alpha"]))
        
        for product in self.species_products:
            self.gc["S2_" + product] = self.gc[product] * self.carbon_species_info.get(product, 0) / self.gc["alpha"] / self.gc["CO2_delta"]

    
    def add_analysis_co2_hydrogenation(self) -> None:
        if self.gc is None:
            logger.error("No GC data available. Run process_gc() first.")
            return None

        delta_moles = self.gc["molar_in"] - self.gc["molar_out"]

        self.gc["H2O"] = 100 - self.gc["sum"]

        # 100% MeOH Selectivity: CO2 + 3H2 -> CH3OH + H2O
        moles_meoh = delta_moles / 4  # Based on reaction stoichiometry
        moles_h2o_meoh = moles_meoh  # Same as MeOH produced
        
        # 100% CO Selectivity: CO2 + H2 -> CO + H2O
        moles_co = delta_moles / 2
        moles_h2o_co = moles_co  # Water produced is same as CO
        
        # Mixed Selectivity (Assuming S_CO = 2S_MeOH)
        alpha_mix = 3  # S_CO : S_MeOH = 2:1
        moles_meoh_mix = delta_moles / (3 + alpha_mix)  # Derived from selectivity definition
        moles_co_mix = alpha_mix * moles_meoh_mix
        moles_h2o_mix = moles_meoh_mix + moles_co_mix

        # Compute final expected concentrations for each case
        self.gc["MeOH_S100%MeOH"] = moles_meoh / self.gc["molar_out"] * 100
        self.gc["CO_S100%MeOH"] = 0
        self.gc["MeOH_S100%CO"] = 0
        self.gc["CO_S100%CO"] = moles_co / self.gc["molar_out"] * 100
        self.gc[f"MeOH_S{(1/(1+alpha_mix))*100:.0f}%MeOH{(1-1/(1+alpha_mix))*100:.0f}%CO"] = moles_meoh_mix / self.gc["molar_out"] * 100
        self.gc[f"CO_S{(1/(1+alpha_mix))*100:.0f}%MeOH{(1-1/(1+alpha_mix))*100:.0f}%CO"] = moles_co_mix / self.gc["molar_out"] * 100
        logger.info("Added theoretical gas concentration analysis for CO2 hydrogenation selectivity scenarios.")

    def mean_group_by_intervals(self, intervals: List[Tuple[float, float]] | List[List[float]] | None = None) -> None:
        if self.gc is None:
            logger.error("No GC data available. Run process_gc() first.")
            return None
        
        if intervals is None:
            intervals = self.intervals
        if intervals is None:
            logger.error("No intervals provided.")
            return None
        
        filtered_gc = self.gc.query("CO2 > 10") 
        
        
        grouped_mean = []
        grouped_std = []
        for interval in intervals:
            if len(interval) != 2:
                logger.error(f"Invalid interval: {interval}")
                continue
            start, end = interval
            gc_mask = (filtered_gc["TOS"] >= start) & (filtered_gc["TOS"] < end)
            gc_mean = filtered_gc[gc_mask].mean(numeric_only=True)
            gc_mean["interval_start"] = start
            gc_mean["interval_end"] = end
            gc_mean["gc_count"] = gc_mask.sum()
            gc_mean.drop("TOS", inplace=True)
            gc_std = filtered_gc[gc_mask].std(numeric_only=True)
            gc_std["interval_start"] = start
            gc_std["interval_end"] = end
            gc_std["gc_count"] = gc_mask.sum()
            gc_std.drop("TOS", inplace=True)

            if self.labview is None:
                group = gc_mean
                grouped_mean.append(group)
                std = gc_std
                grouped_std.append(std)
                continue

            lv_mask = (self.labview["TOS"] >= start) & (self.labview["TOS"] < end)
            lv_mean = self.labview[lv_mask].mean(numeric_only=True)
            lv_mean["lv_count"] = lv_mask.sum()
            lv_mean.drop("TOS", inplace=True)
            lv_std = self.labview[lv_mask].std(numeric_only=True)
            lv_std["lv_count"] = lv_mask.sum()
            lv_std.drop("TOS", inplace=True)


            mean = pd.concat([gc_mean, lv_mean])
            std = pd.concat([gc_std, lv_std])
            grouped_mean.append(mean)
            grouped_std.append(std)
        
        self.grouped_mean = pd.DataFrame(grouped_mean)
        self.grouped_mean.insert(0, "interval_start", self.grouped_mean.pop("interval_start"))
        self.grouped_mean.insert(1, "interval_end", self.grouped_mean.pop("interval_end"))
        self.grouped_mean.insert(2, "gc_count", self.grouped_mean.pop("gc_count"))
        self.grouped_mean.insert(2, "lv_count", self.grouped_mean.pop("lv_count"))

        self.grouped_std = pd.DataFrame(grouped_std)
        self.grouped_std.insert(0, "interval_start", self.grouped_std.pop("interval_start"))
        self.grouped_std.insert(1, "interval_end", self.grouped_std.pop("interval_end"))
        self.grouped_std.insert(2, "gc_count", self.grouped_std.pop("gc_count"))
        self.grouped_std.insert(2, "lv_count", self.grouped_std.pop("lv_count"))
        logger.info("Grouped mean and standard deviation by intervals.")
        return None