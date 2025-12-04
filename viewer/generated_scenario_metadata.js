/**
 * Auto-generated Scenario Metadata
 * DO NOT EDIT MANUALLY
 * 
 * Regenerate with: python viewer/generate_scenario_metadata.py
 */

export const ScenarioSchemas = {
    "binary_star": {
        "params": {
            "mass": {
                "type": "float",
                "default": 5.0,
                "min": 0.1,
                "max": 100.0
            },
            "separation": {
                "type": "float",
                "default": 10.0,
                "min": 1.0,
                "max": 100.0
            },
            "G": {
                "type": "float",
                "default": 2.0,
                "min": 0.1,
                "max": 100.0
            },
            "dt": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 1.0
            },
            "dim": {
                "type": "int",
                "default": 2,
                "allowed": [
                    2,
                    3
                ]
            }
        },
        "presets": {
            "wide_binary": {
                "separation": 20.0,
                "dt": 0.05
            },
            "tight_binary": {
                "separation": 5.0,
                "dt": 0.01,
                "G": 5.0
            },
            "eccentric_binary": {
                "separation": 15.0
            }
        }
    },
    "bubble_collapse": {
        "params": {
            "N": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 5000
            },
            "radius": {
                "type": "float",
                "default": 10.0,
                "min": 1.0,
                "max": 100.0
            },
            "initial_asymmetry": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0
            },
            "dim": {
                "type": "int",
                "default": 2,
                "allowed": [
                    2,
                    3
                ]
            }
        },
        "presets": {
            "slow": {
                "dt": 0.005,
                "G": 0.5
            },
            "rapid": {
                "dt": 0.05,
                "G": 8.0
            },
            "symmetric": {
                "initial_asymmetry": 0.0,
                "dt": 0.01
            },
            "fragmented": {
                "initial_asymmetry": 0.3,
                "N": 200
            },
            "symmetric_3d": {
                "dim": 3,
                "N": 500,
                "dt": 0.01
            },
            "fragmented_3d": {
                "dim": 3,
                "N": 1000,
                "initial_asymmetry": 0.4
            },
            "shell_thickening": {
                "dim": 3,
                "N": 2000,
                "radius": 20.0,
                "initial_asymmetry": 0.1
            }
        }
    },
    "bulk_ring": {
        "params": {
            "N": {
                "type": "int",
                "default": 64,
                "min": 1,
                "max": 128,
                "description": "Number of entities in the ring"
            },
            "radius": {
                "type": "float",
                "default": 8.0,
                "min": 1.0,
                "max": 50.0,
                "description": "Orbital radius of the ring"
            },
            "speed": {
                "type": "float",
                "default": 0.8,
                "min": 0.1,
                "max": 5.0,
                "description": "Tangential velocity"
            },
            "mass": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 10.0,
                "description": "Mass per entity"
            }
        },
        "presets": {
            "stable_ring": {
                "N": 64,
                "radius": 12.0,
                "speed": 0.8,
                "mass": 1.0
            },
            "wide_ring": {
                "N": 128,
                "radius": 20.0,
                "speed": 0.6
            },
            "chaos_ring": {
                "N": 50,
                "radius": 5.0,
                "speed": 1.2,
                "mass": 0.5
            }
        }
    },
    "mini_solar": {
        "params": {
            "sun_mass": {
                "type": "float",
                "default": 20.0,
                "min": 1.0,
                "max": 1000.0
            },
            "num_planets": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10
            },
            "G": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 100.0
            },
            "dt": {
                "type": "float",
                "default": 0.03,
                "min": 0.001,
                "max": 1.0
            },
            "dim": {
                "type": "int",
                "default": 2,
                "allowed": [
                    2,
                    3
                ]
            }
        },
        "presets": {
            "three_planet": {
                "num_planets": 3
            },
            "compact_system": {
                "num_planets": 5,
                "G": 2.0
            },
            "outer_belt": {
                "num_planets": 8,
                "sun_mass": 50.0
            }
        }
    },
    "mobius_walk": {
        "params": {
            "speed": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 10.0
            }
        },
        "presets": {
            "slow": {
                "dt": 0.1,
                "speed": 1.0
            },
            "fast": {
                "dt": 0.02,
                "speed": 5.0
            },
            "wrap-test": {
                "speed": 2.0,
                "topology_type": 1
            }
        }
    },
    "random_nbody": {
        "params": {
            "N": {
                "type": "int",
                "default": 25,
                "min": 1,
                "max": 5000
            },
            "radius": {
                "type": "float",
                "default": 10.0,
                "min": 1.0,
                "max": 100.0
            },
            "dim": {
                "type": "int",
                "default": 3,
                "allowed": [
                    2,
                    3
                ]
            }
        },
        "presets": {
            "small_cluster": {
                "N": 200
            },
            "medium_cluster": {
                "N": 1000
            },
            "large_cluster": {
                "N": 2500
            },
            "fast": {
                "dt": 0.05,
                "G": 1.0,
                "N": 40,
                "radius": 12.0
            },
            "dense": {
                "N": 200,
                "radius": 5.0,
                "G": 5.0
            },
            "chaotic": {
                "dt": 0.1,
                "G": 15.0,
                "N": 75
            },
            "gentle": {
                "dt": 0.02,
                "G": 0.5,
                "radius": 20.0
            }
        }
    },
    "random_nbody_3d": {
        "params": {
            "N": {
                "type": "int",
                "default": 300,
                "min": 10,
                "max": 10000
            },
            "radius": {
                "type": "float",
                "default": 20.0,
                "min": 1.0,
                "max": 200.0
            },
            "distribution": {
                "type": "str",
                "default": "spherical",
                "allowed": [
                    "spherical",
                    "gaussian",
                    "disc"
                ]
            },
            "rotation": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 5.0
            },
            "expansion": {
                "type": "float",
                "default": 0.0,
                "min": -5.0,
                "max": 5.0
            },
            "G": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 100.0
            },
            "dt": {
                "type": "float",
                "default": 0.01,
                "min": 0.001,
                "max": 1.0
            },
            "dim": {
                "type": "int",
                "default": 3,
                "allowed": [
                    3
                ]
            }
        },
        "presets": {
            "galaxy_disc": {
                "distribution": "disc",
                "N": 500,
                "rotation": 1.0,
                "radius": 30.0,
                "G": 2.0
            },
            "globular_cluster": {
                "distribution": "gaussian",
                "N": 400,
                "radius": 15.0,
                "rotation": 0.2
            },
            "expanding_cloud": {
                "distribution": "spherical",
                "N": 300,
                "expansion": 2.0,
                "G": 0.5
            },
            "high_energy_chaos": {
                "distribution": "spherical",
                "N": 200,
                "radius": 10.0,
                "expansion": 0.0,
                "rotation": 0.0,
                "G": 10.0,
                "dt": 0.005
            }
        }
    },
    "stellar_trio": {
        "params": {
            "dim": {
                "type": "int",
                "default": 2,
                "allowed": [
                    2,
                    3
                ]
            }
        },
        "presets": {
            "stable_figure8": {
                "dim": 2
            },
            "chaotic_3d": {
                "dim": 3
            },
            "stable_plane_3d": {
                "dim": 3
            },
            "inverted_orbits": {
                "dim": 3
            }
        }
    },
    "vortex_sheet": {
        "params": {
            "N": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 5000
            },
            "viscosity": {
                "type": "float",
                "default": 0.1,
                "min": 0.0,
                "max": 1.0
            },
            "dim": {
                "type": "int",
                "default": 2,
                "allowed": [
                    2
                ]
            }
        },
        "presets": {
            "laminar": {
                "dt": 0.01,
                "viscosity": 0.3
            },
            "turbulent": {
                "dt": 0.002,
                "viscosity": 0.01
            },
            "shock": {
                "dt": 0.0005,
                "viscosity": 0.001
            },
            "colliding_streams": {
                "dt": 0.005,
                "N": 200
            },
            "shear_layer": {
                "dt": 0.002,
                "N": 300,
                "viscosity": 0.05
            }
        }
    }
};
