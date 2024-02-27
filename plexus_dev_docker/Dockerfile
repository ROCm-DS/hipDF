FROM rocm/dev-ubuntu-22.04:6.0.2-complete

SHELL ["/bin/bash", "-c"]

ARG GITHUB_USER
ARG GITHUB_PASS

#install necessary system packages
RUN apt-get update \
    && apt-get install -y wget git vim \
    && apt-get install -y rocthrust hipcub rocm-hip-libraries

#install conda
ARG ANACONDA_PYTHON_VERSION
ENV ANACONDA_PYTHON_VERSION=3.10
ENV PATH /opt/conda/envs/cudf_dev/bin:/opt/conda/bin:$PATH
COPY install_conda.sh /tmp
RUN cd /tmp && bash install_conda.sh && rm /tmp/install_conda.sh

ENV CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake

#prepare build folder
RUN mkdir -p /build
WORKDIR  /build

# #build RMM from source
RUN git clone https://${GITHUB_USER}:${GITHUB_PASS}@github.com/AMD-AI/RMM.git && cd RMM && git checkout rocm-6.0/develop-23.12 
RUN cd /build/RMM && conda env create --name rmm_dev --file conda/environments/all_rocm_arch-x86_64.yaml 
RUN . /opt/conda/etc/profile.d/conda.sh &&  conda activate rmm_dev \
   && export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake \ 
   && cd /build/RMM \
   && python3 -m pip install -r conda/environments/rocm-requirements.txt \
   && ./build.sh rmm  

#build cupy from source
RUN git clone https://${GITHUB_USER}:${GITHUB_PASS}@github.com/AMD-AI/cupy.git && cd cupy \
   && git checkout cuda_array_interface_dev_rocm_6.0 \
   && git submodule update --init
COPY cupy_dev.yaml /build/
RUN conda env create -n cupy_dev -f cupy_dev.yaml
RUN . /opt/conda/etc/profile.d/conda.sh &&  conda activate cupy_dev \
   && export CUPY_INSTALL_USE_HIP=1 \
   && export ROCM_HOME=/opt/rocm \
   && export HCC_AMDGPU_TARGET=gfx90a \
   && cd /build/cupy \
   && python3 setup.py bdist_wheel

#clone hipdf
RUN git clone https://${GITHUB_USER}:${GITHUB_PASS}@github.com/AMD-AI/hipdf.git && cd hipdf && git checkout dev

#prepare conda environment for hipdf build
RUN cd /build/hipdf && conda env create --name cudf_dev --file conda/environments/all_rocm_arch-x86_64.yaml 

#patch numba
RUN . /opt/conda/etc/profile.d/conda.sh &&  conda activate cudf_dev \
 && git clone https://${GITHUB_USER}:${GITHUB_PASS}@github.com/AMD-AI/rocnumba2.git \
 && cd rocnumba2 && git checkout feat/rocm-kickoff && bash patch-active-conda-env.sh
#
#add hip-python, RMM and cupy custom build to conda environment
RUN . /opt/conda/etc/profile.d/conda.sh &&  conda activate cudf_dev \
  && pip install -r /build/hipdf/conda/environments/rocm-requirements.txt \
  && pip install /build/RMM/python/dist/rmm_rocm-23.12.0.0-cp310-cp310-linux_x86_64.whl \
  && pip install /build/cupy/dist/cupy-13.0.0b1-cp310-cp310-linux_x86_64.whl 

#build hipdf python package
RUN . /opt/conda/etc/profile.d/conda.sh \
  && conda activate cudf_dev \
  && export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake \ 
  && cd /build/hipdf && bash build_hip.sh hipdf \
  && pip install /build/hipdf/python/cudf/dist/cudf-23.10.0.0-cp310-cp310-linux_x86_64.whl
