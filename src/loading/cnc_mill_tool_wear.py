"""
Module implementing a Dataloader for the CNC Mill Tool Wear dataset.

The dataset can be found under: 
https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill
https://github.com/Nhianh03/cnc_mill_tool_wear

Dataset description:
--------------------
General Information:
The dataset consists of 18 experiments (= samples). Each experiment is one machining process.
Process parameters like feed rate, tool condition and clamping pressure are varied.
The data are sampled at 10 Hz.

Features:
The following sensors are used:
- X1_ActualPosition: Actual x position of part [mm]
- X1_ActualVelocity: Actual x velocity of part [mm/s]
- X1_ActualAcceleration: Actual x acceleration of part [mm/s^2]
- X1_CommandPosition: Reference x position of part [mm]
- X1_CommandVelocity: Reference x velocity of part [mm/s]
- X1_CommandAcceleration: Reference x acceleration of part [mm/s^2]
- X1_CurrentFeedback: Current [A]
- X1_DCBusVoltage: Voltage [V]
- X1_OutputCurrent: Current [A]
- X1_OutputVoltage: Voltage [V]
- X1_OutputPower: Output power [kW]

- Y1_ActualPosition: Actual y position of part [mm]
- Y1_ActualVelocity: Actual y velocity of part [mm/s]
- Y1_ActualAcceleration: Actual y acceleration of part [mm/s^2]
- Y1_CommandPosition: Reference y position of part [mm]
- Y1_CommandVelocity: Reference y velocity of part [mm/s]
- Y1_CommandAcceleration: Reference y acceleration of part [mm/s^2]
- Y1_CurrentFeedback: Current [A]
- Y1_DCBusVoltage: Voltage [V]
- Y1_OutputCurrent: Current [A]
- Y1_OutputVoltage: Voltage [V]
- Y1_OutputPower: Output power [kW]

- Z1_ActualPosition: Actual z position of part [mm]
- Z1_ActualVelocity: Actual z velocity of part [mm/s]
- Z1_ActualAcceleration: Actual z acceleration of part [mm/s^2]
- Z1_CommandPosition: Reference z position of part [mm]
- Z1_CommandVelocity: Reference z velocity of part [mm/s]
- Z1_CommandAcceleration: Reference z acceleration of part [mm/s^2]
- Z1_CurrentFeedback: Current [A]
- Z1_DCBusVoltage: Voltage [V]
- Z1_OutputCurrent: Current [A]
- Z1_OutputVoltage: Voltage [V]

- S1_ActualPosition: Actual position of spindle [mm]
- S1_ActualVelocity: Actual velocity of spindle [mm/s]
- S1_ActualAcceleration: Actual acceleration of spindle [mm/s^2]
- S1_CommandPosition: Reference position of spindle [mm]
- S1_CommandVelocity: Reference velocity of spindle [mm/s]
- S1_CommandAcceleration: Reference acceleration of spindle [mm/s^2]
- S1_CurrentFeedback: Current [A]
- S1_DCBusVoltage: Voltage [V]
- S1_OutputCurrent: Current [A]
- S1_OutputVoltage: Voltage [V]
- S1_OutputPower: Current [A]
- S1_SystemInertia: Torque inertia [kg*m^2]

- M1_CURRENT_FEEDRATE: Instantaneous feed rate of spindle

Target Variables:
The dataset has 2 different target variables:
- Tool condition: 2 classes [worn / unworn]
- Clamp pressure: 3 classes [2.5 / 3 / 4 bar]

"""

from __future__ import annotations

import os
from pathlib import Path

import fsspec
import pandas as pd
from tqdm import tqdm

class Dataloader:
    """
    Data loader for the CNC Mill Tool Wear dataset.

    The dataset can be found under:
    https://github.com/Nhianh03/cnc_mill_tool_wear

    Dataset description:
    --------------------
    General Information:
    The dataset consists of 18 experiments (= samples). Each experiment is one machining process.
    Process parameters like feed rate, tool condition and clamping pressure are varied.
    The data are sampled at 10 Hz.

    Features:
    The following sensors are used:
    - X1_ActualPosition: Actual x position of part [mm]
    - X1_ActualVelocity: Actual x velocity of part [mm/s]
    - X1_ActualAcceleration: Actual x acceleration of part [mm/s^2]
    - X1_CommandPosition: Reference x position of part [mm]
    - X1_CommandVelocity: Reference x velocity of part [mm/s]
    - X1_CommandAcceleration: Reference x acceleration of part [mm/s^2]
    - X1_CurrentFeedback: Current [A]
    - X1_DCBusVoltage: Voltage [V]
    - X1_OutputCurrent: Current [A]
    - X1_OutputVoltage: Voltage [V]
    - X1_OutputPower: Output power [kW]

    - Y1_ActualPosition: Actual y position of part [mm]
    - Y1_ActualVelocity: Actual y velocity of part [mm/s]
    - Y1_ActualAcceleration: Actual y acceleration of part [mm/s^2]
    - Y1_CommandPosition: Reference y position of part [mm]
    - Y1_CommandVelocity: Reference y velocity of part [mm/s]
    - Y1_CommandAcceleration: Reference y acceleration of part [mm/s^2]
    - Y1_CurrentFeedback: Current [A]
    - Y1_DCBusVoltage: Voltage [V]
    - Y1_OutputCurrent: Current [A]
    - Y1_OutputVoltage: Voltage [V]
    - Y1_OutputPower: Output power [kW]

    - Z1_ActualPosition: Actual z position of part [mm]
    - Z1_ActualVelocity: Actual z velocity of part [mm/s]
    - Z1_ActualAcceleration: Actual z acceleration of part [mm/s^2]
    - Z1_CommandPosition: Reference z position of part [mm]
    - Z1_CommandVelocity: Reference z velocity of part [mm/s]
    - Z1_CommandAcceleration: Reference z acceleration of part [mm/s^2]
    - Z1_CurrentFeedback: Current [A]
    - Z1_DCBusVoltage: Voltage [V]
    - Z1_OutputCurrent: Current [A]
    - Z1_OutputVoltage: Voltage [V]

    - S1_ActualPosition: Actual position of spindle [mm]
    - S1_ActualVelocity: Actual velocity of spindle [mm/s]
    - S1_ActualAcceleration: Actual acceleration of spindle [mm/s^2]
    - S1_CommandPosition: Reference position of spindle [mm]
    - S1_CommandVelocity: Reference velocity of spindle [mm/s]
    - S1_CommandAcceleration: Reference acceleration of spindle [mm/s^2]
    - S1_CurrentFeedback: Current [A]
    - S1_DCBusVoltage: Voltage [V]
    - S1_OutputCurrent: Current [A]
    - S1_OutputVoltage: Voltage [V]
    - S1_OutputPower: Current [A]
    - S1_SystemInertia: Torque inertia [kg*m^2]

    - M1_CURRENT_FEEDRATE: Instantaneous feed rate of spindle

    Target Variables:
    The dataset has 2 different target variables:
    - Tool condition: 2 classes [worn / unworn]
    - Clamp pressure: 3 classes [2.5 / 3 / 4 bar]

    Attributes
    ----------
    drop_feature_columns : list[str]
        A list of columns to drop from the features.
    target_columns : list[str]
        A list of columns to use as target variables.
    target_file : str
        The file containing the target variables.

    Methods
    -------
    load_dataset(dir_path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]
        Load the CNC Mill Tool Wear dataset.
    """

    drop_feature_columns = [
        "M1_CURRENT_PROGRAM_NUMBER",
        "M1_sequence_number",
        "Machining_Process",
    ]
    target_columns = ["clamp_pressure", "tool_condition"]
    target_file = "train.csv"
    url = "https://github.com/Nhianh03/cnc_mill_tool_wear"

    def load_dataset(
        self, dir_path: Path
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load the CNC Mill Tool Wear dataset. This method will download
        the dataset if it is not already downloaded.

        Parameters
        ----------
        dir_path : Path
            Path, where the dataset will be located. If this directory is empty,
            the dataset will be downloaded to this directory. If the directory does not exist,
            it will be created.

        Returns
        -------
        tuple[dict[str, pd.DataFrame], pd.DataFrame]
            A tuple containing the features and target data.
        """
        # Check if the directory already exists, if not, create it
        dir_path = Path(dir_path)
        if not dir_path.exists():
            os.makedirs(dir_path)

        # Check if the directory is empty --> download necessary
        if not os.listdir(dir_path):
            self._download_dataset(dir_path)

        # Load the dataset
        features, targets = self._load_from_dir(dir_path)

        return features, targets

    def _download_dataset(self, dir_path: Path) -> None:
        """
        Download the CNC Mill Tool Wear dataset from the GitHub repository.

        Parameters
        ----------
        dir_path : Path
            The directory to save the dataset to.
        """
        print(f"Downloading CNC Mill Tool Wear dataset to {dir_path}...")

        # Download the features
        destination = Path(dir_path) / "features"
        destination.mkdir(parents=True, exist_ok=True)
        fs = fsspec.filesystem("github", org="Nhianh03", repo="cnc_mill_tool_wear")
        fs.get(fs.ls("data/"), destination.as_posix())

        # Download targets
        destination = Path(dir_path) / "train.csv"
        fs.get("train.csv", destination.as_posix())

    def _load_from_dir(
        self, dir_path: Path
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load the CNC Mill Tool Wear dataset from a directory.

        Parameters
        ----------
        dir_path : Path
            The directory to load the dataset from.

        Returns
        -------
        tuple[dict[str, pd.DataFrame], pd.DataFrame]
            A tuple containing the features and target data.
        """
        print(f"Loading CNC Mill Tool Wear dataset from {dir_path}...")
        experiments = self._load_experiments(dir_path)
        features = self._extract_features_from_experiments(experiments)
        targets = self._load_targets(dir_path / self.target_file)

        return features, targets

    def _load_experiments(self, dir_path: Path) -> dict[str, pd.DataFrame]:
        """
        Load the CNC Mill Tool Wear dataset from a directory.

        Parameters
        ----------
        dir_path : Path
            The directory to load the dataset from.

        Returns
        -------
        dict[str, pd.DataFrame]
            A dictionary containing the features data.
        """
        experiments = {}
        for file in (dir_path / "features").iterdir():
            experiments[file.stem] = pd.read_csv(file)
            # Drop unnecessary columns
            experiments[file.stem].drop(columns=self.drop_feature_columns, inplace=True)

        return experiments

    def _extract_features_from_experiments(
        self, experiments: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Extract the features from the experiments.

        Parameters
        ----------
        experiments : dict[str, pd.DataFrame]
            A dictionary containing the experiments data.

        Returns
        -------
        dict[str, pd.DataFrame]
            A dictionary containing the features data.
        """
        # For every column in features: Concat all dataframes into one and add a measurement column
        sensor_names = experiments["experiment_01"].columns.values
        features = {}
        for sensor in tqdm(sensor_names):
            sensor_df = pd.DataFrame()
            for idx, data in enumerate(experiments.values()):
                temp_df = data[[sensor]].copy()
                # Rename sensor column to `value`
                temp_df.columns = ["value"]
                # Add measurement index
                temp_df["measurement_idx"] = str(idx + 1)
                sensor_df = pd.concat([sensor_df, temp_df], axis=0)
            assert len(sensor_df) == 25286
            features[sensor] = sensor_df

        return features

    def _load_targets(self, target_file: Path) -> pd.DataFrame:
        """
        Load the target variables from a file.

        Parameters
        ----------
        target_file : Path
            The file containing the target variables.

        Returns
        -------
        pd.DataFrame
            The target variables as a DataFrame.
        """
        targets = pd.read_csv(target_file, usecols=self.target_columns)
        assert len(targets) == 18

        return targets
