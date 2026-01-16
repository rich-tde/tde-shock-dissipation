#################################################################################
# GLOBALS                                                                       #
#################################################################################

RICH_PROBLEM := ShockTube
PL ?= 1
DL ?= 1
PR ?= 0.1
DR ?= 0.125

LMOD_ENV := new_rich_build			# This will be called by module restore $(LMOD_ENV)
CONDA_ENV := richanalysis
RICH_PATH := /home/hey4/RICH-fwrk
DATA_PATH := /home/hey4/rich_tde/data/raw/ShockTubeIdealGas
OUTPUT_PATH := $(DATA_PATH)/PL$(PL)PR$(PR)DL$(DL)DR$(DR)

MPI ?= 0

ifeq ($(MPI),1)
MPI_BUILD_SUFFIX := --mpi
else
MPI_BUILD_SUFFIX :=
endif

.DEFAULT_GOAL := one

.PHONY: one
one: $(OUTPUT_PATH)/snap_final.h5

.PHONY: all
all:
	$(MAKE) all_ics
	$(MAKE) run_all		#recursive make to make sure IC_DIRS are listed after all_ics are run

IC_DIRS = $(shell ls -d $(DATA_PATH)/PL*PR*DL*DR* 2>/dev/null)
ALL_SNAPS = $(IC_DIRS:%=%/snap_final.h5)

.PHONY: run_all
run_all: $(ALL_SNAPS)

#################################################################################
# MAIN                                                                          #
#################################################################################

## Single run, only defined if not make all
IS_ALL := $(filter all,$(MAKECMDGOALS))
ifneq ($(IS_ALL),all)
ifeq ($(MPI),1)
$(OUTPUT_PATH)/snap_final.h5: run_mpi.sh $(RICH_PATH)/build/rich $(OUTPUT_PATH)/leftpressure.txt \
  $(OUTPUT_PATH)/leftdensity.txt $(OUTPUT_PATH)/rightpressure.txt $(OUTPUT_PATH)/rightdensity.txt
	cd $(@D) && sbatch $(CURDIR)/run_mpi.sh
else
$(OUTPUT_PATH)/snap_final.h5: run_serial.sh $(RICH_PATH)/build/rich $(OUTPUT_PATH)/leftpressure.txt \
  $(OUTPUT_PATH)/leftdensity.txt $(OUTPUT_PATH)/rightpressure.txt $(OUTPUT_PATH)/rightdensity.txt
	cd $(@D) && $(CURDIR)/run_serial.sh
endif
endif

ifeq ($(MPI),1)
$(DATA_PATH)/%/snap_final.h5: run_mpi.sh $(RICH_PATH)/build/rich $(DATA_PATH)/%/leftpressure.txt \
  $(DATA_PATH)/%/leftdensity.txt $(DATA_PATH)/%/rightpressure.txt $(DATA_PATH)/%/rightdensity.txt
	cd $(@D) && sbatch $(CURDIR)/run_mpi.sh
else
$(DATA_PATH)/%/snap_final.h5: run_serial.sh $(RICH_PATH)/build/rich $(DATA_PATH)/%/leftpressure.txt \
  $(DATA_PATH)/%/leftdensity.txt $(DATA_PATH)/%/rightpressure.txt $(DATA_PATH)/%/rightdensity.txt
	cd $(@D) && $(CURDIR)/run_serial.sh
endif

$(RICH_PATH)/build/rich: $(RICH_PATH)/runs/ShockTube/main.cpp build.sh .mpi_stamp_$(MPI)
	./build.sh

ifeq ($(MPI),1)
run_mpi.sh:
	printf '%s\n' \
		'#!/bin/bash -l' \
		'#SBATCH --partition=gpu_strw' \
		'#SBATCH --account=gpu_strw' \
		'#SBATCH --job-name=shocktubes' \
		'#SBATCH --time=0-01:00:00 # max 7 days' \
		'#SBATCH --output=/home/hey4/RICH/jobs/logs/%j_%x.out   # %x: Job name, %j: Job id' \
		'#SBATCH --nodes=1' \
		'#SBATCH --ntasks=12	# max 48 per node for gpu_strw' \
		'#SBATCH --cpus-per-task=1' \
		'#SBATCH --mem=61G	    # max 61G per node for gpu_strw' \
		'#SBATCH --mail-user="yujiehe@strw.leidenuniv.nl"' \
		'#SBATCH --mail-type="ALL"' \
		'echo "Date: $$(date)"' \
		'echo "Batch script:"' \
		'cat "$$0"' \
		'echo "==============================================="' \
		'module purge && module restore $(LMOD_ENV)' \
		'srun $(RICH_PATH)/build/rich' \
	> $@
else
run_serial.sh:
	printf '%s\n' \
		'#!/bin/bash -l' \
		'module purge && module restore $(LMOD_ENV)' \
		'$(RICH_PATH)/build/rich' \
	> $@
	chmod +x $@
endif

build.sh:
	printf '%s\n' \
		'#!/bin/bash -l' \
		'rm -r $(RICH_PATH)/build' \
		'module purge && module restore $(LMOD_ENV)' \
		'$(RICH_PATH)/config.py --problem=$(RICH_PROBLEM) $(MPI_BUILD_SUFFIX)' \
		'$(MAKE) -C $(RICH_PATH)' \
	> $@
	chmod +x $@

ifeq ($(MPI),1)
.mpi_stamp_1:
	rm -f .mpi_stamp_0
	touch $@
else
.mpi_stamp_0:
	rm -f .mpi_stamp_1
	touch $@
endif	

.PHONY: all_ics
all_ics:
	source ~/.bashrc && conda activate $(CONDA_ENV) && python ics.py $(DATA_PATH) -n 10

$(OUTPUT_PATH)/leftpressure.txt:
	mkdir -p $(@D) && echo $(PL) > $@

$(OUTPUT_PATH)/leftdensity.txt:
	mkdir -p $(@D) && echo $(DL) > $@

$(OUTPUT_PATH)/rightpressure.txt:
	mkdir -p $(@D) && echo $(PR) > $@

$(OUTPUT_PATH)/rightdensity.txt:
	mkdir -p $(@D) && echo $(DR) > $@