import logging
from datetime import datetime
from pathlib import Path

from aind_behavior_vr_foraging_packaging.acquisition import AcquisitionProcessor
from aind_behavior_vr_foraging_packaging.nwb_file import NwbSession, _AindDataSchemaJson
from aind_behavior_vr_foraging_packaging.processing import (
    CreateProcessingModuleProcessor,
    LicksProcessor,
    PositionAndVelocityProcessor,
    SniffingProcessor,
    TrialTableProcessor,
)
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, ProcessStage
from aind_data_schema_models.process_names import ProcessName
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)
VERSION = "9.0"
GITHUB_URL = "https://github.com/AllenNeuralDynamics/aind-vr-foraging-primary-data-nwb-packaging.git"


class VRForagingSettings(BaseSettings, cli_parse_args=True):
    """Settings for VR Foraging Primary Data NWB Packaging."""

    input_directory: Path = Field(
        default=Path("/data/"), description="Directory where data is"
    )
    output_directory: Path = Field(
        default=Path("/results/"), description="Output directory"
    )


class LocalNwbSession(NwbSession):
    """NwbSession that reads aind-data-schema metadata from the attached asset on disk."""

    def _get_aind_data_schema_json(self) -> _AindDataSchemaJson:
        return _AindDataSchemaJson.from_root_path(self.root_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    settings = VRForagingSettings()
    start_process_time = datetime.now()

    primary_data_paths = tuple(
        p for p in settings.input_directory.glob("*") if p.is_dir()
    )
    if not primary_data_paths:
        raise FileNotFoundError("No primary data asset attached")
    if len(primary_data_paths) > 1:
        raise ValueError(
            "Multiple primary data assets attached. Only single asset needed"
        )
    primary_data_path = primary_data_paths[0]

    session = LocalNwbSession(primary_data_path)
    logger.info(
        "Packaging %s (dataset version %s)",
        primary_data_path.name,
        session.dataset_version,
    )

    session.run(
        AcquisitionProcessor(session.dataset),
        CreateProcessingModuleProcessor(session.dataset),
        PositionAndVelocityProcessor(session.dataset),
        SniffingProcessor(session.dataset),
        LicksProcessor(session.dataset),
        TrialTableProcessor(session.dataset),
    )

    nwb_result_path = settings.output_directory / "behavior.nwb.zarr"
    logger.info(
        "Successfully finished nwb packaging. Writing to disk at %s as zarr",
        nwb_result_path,
    )
    session.write_nwb_zarr(nwb_result_path)

    data_process = DataProcess(
        start_date_time=start_process_time,
        end_date_time=datetime.now(),
        stage=ProcessStage.PROCESSING,
        process_type=ProcessName.PIPELINE,
        experimenters=["Arjun Sridhar"],
        code=Code(url=GITHUB_URL, version=VERSION),
        output_parameters={},
        notes=f"Run with dataset version: {session.dataset_version}",
    )
    with open(settings.output_directory / "data_process.json", "w") as f:
        f.write(data_process.model_dump_json(indent=4))
