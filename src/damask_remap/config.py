from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GridConfig(BaseModel):
    cells: int = Field(default=16, gt=0, le=32)
    size: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    origin: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class MaterialConfig(BaseModel):
    phase: str = "Aluminium"
    seed: int = 42


class LoadConfig(BaseModel):
    mode: Literal["rolling"] = "rolling"
    solver: Literal["basic"] = "basic"
    t: int = Field(default=100, gt=0)
    N: int = Field(default=200, gt=0)
    f_out: int = Field(default=10, gt=0)
    rate: float = Field(default=1e-3, gt=0)

    @model_validator(mode="after")
    def final_increment_saved(self):
        if self.N % self.f_out != 0:
            raise ValueError(f"N ({self.N}) must be a multiple of f_out ({self.f_out})")
        return self


class RunConfig(BaseModel):
    name: str = "input_files"
    grid: GridConfig = Field(default_factory=GridConfig)
    load: LoadConfig = Field(default_factory=LoadConfig)
    material: MaterialConfig = Field(default_factory=MaterialConfig)
