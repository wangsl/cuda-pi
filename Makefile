.DEFAULT: .f .for .c .C .cpp .cc .f90
.SUFFIXES: .f .for .c .C .cpp .cc .f90

O = .

F77 = ifort
F90 = ifort
CC = icc
CCC = icpc
NVCC = nvcc

TYPEDEF = #-DFLOAT

NVCCINCLUDE = -I$(CUDA_ROOT)/samples/common/inc

CFLAGS = $(NVCCINCLUDE) $(TYPEDEF)

CFLAGS = $(NVCCINCLUDE) $(TYPEDEF)

FFLAGS = $(TYPEDEF)

NVCCFLAGS = -gencode arch=compute_20,code=sm_20 \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_37,code=sm_37 -use_fast_math -O3 \
	$(NVCCINCLUDE) $(TYPEDEF) -Xcompiler=\"-fPIC -pthread -fexceptions -m64\"

Link = $(CCC) $(CFLAGS)

LIBS = -L$(CUDA_LIB) -lcudart  -lifcore

EXENAME = t1

OBJS =	$(O)/main.o  $(O)/pi.o  $(O)/pi-omp.o  $(O)/mainF.o  $(O)/gpuprop.o

$(EXENAME) : $(OBJS) 
	$(Link) -o $(EXENAME) $(OBJS)  $(LIBS) 

$(O)/%.o: %.c
	cd $(O) ; $(CC) $(CFLAGS) -c $<
$(O)/%.o: %.cc
	cd $(O) ; $(CCC) $(CFLAGS) -c $<
$(O)/%.o: %.cpp
	cd $(O) ; $(CCC) $(CFLAGS) -c $<
$(O)/%.o: %.C
	cd $(O) ; $(CCC) $(CFLAGS) -c $<
$(O)/%.o: %.F
	cd $(O) ; $(F77) $(FFLAGS) -c $<
$(O)/%.o: %.for
	cd $(O) ; $(F77) $(FFLAGS) -c $<
$(O)/%.o: %.f90
	cd $(O) ; $(F90) $(FFLAGS) -c $<
$(O)/%.o: %.cu
	cd $(O) ; $(NVCC) $(NVCCFLAGS) -c $<

dat: 
	rm -f *.dat
backup:

	rm -f *~
clobber:
	rm -f $(EXENAME).exe

clean:
	rm -f *.o *.dat *~ *.exe *.exe.* $(EXENAME) 
	rm -f *.pc *.pcl *.*i *.mod depend *.linkinfo

depend :
	g++ $(NVCCINCLUDE) -MM *.[cC]  | perl dep.pl > $@
	nvcc $(NVCCFLAGS) -M *.cu | perl dep.pl >> $@

include depend
