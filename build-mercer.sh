#!/bin/bash

# $Id: build-nyu.sh 38 2013-06-19 19:12:11Z wangsl2001@gmail.com $

#export BUILD_WRAPPER_SCRIPT=
#export SPECIAL_RULES_FUNCTION=

#export SOURCE_CODE_LIST_WITH_INTEL_COMPILERS=
#export SOURCE_CODE_LIST_WITH_GNU_COMPILERS=
#export REGULAR_EXPRESSION_LIST_FOR_SOURCE_CODE_WITH_INTEL_COMPILERS=
#export REGULAR_EXPRESSION_LIST_FOR_SOURCE_CODE_WITH_GNU_COMPILERS=
#export INVALID_FLAGS_FOR_INTEL_COMPILERS=
#export INVALID_FLAGS_FOR_GNU_COMPILERS=
#export OPTIMIZATION_FLAGS=
#export OPTIMIZATION_FLAGS_FOR_INTEL_COMPILERS=
#export OPTIMIZATION_FLAGS_FOR_INTEL_FORTRAN_COMPILERS=
#export OPTIMIZATION_FLAGS_FOR_GNU_COMPILERS=
#export OPTIMIZATION_FLAGS_FOR_GNU_FORTRAN_COMPILERS=
#export INCLUDE_FLAGS=
#export INCLUDE_FLAGS_FOR_INTEL_COMPILERS=
#export INCLUDE_FLAGS_FOR_INTEL_FORTRAN_COMPILERS=
#export INCLUDE_FLAGS_FOR_GNU_COMPILERS=
#export INCLUDE_FLAGS_FOR_GNU_FORTRAN_COMPILERS=
#export LINK_FLAGS=
#export LINK_FLAGS_FOR_INTEL_COMPILERS="
#export LINK_FLAGS_FOR_GNU_COMPILERS="-fopenmp"
#export EXTRA_OBJECT_FILE_AND_LIBRARY_LIST=
#export STRING_MACROS=
#export FUNCTION_MACROS=
#export INTEL_MPI_BIN_PATH=
#export GNU_MPI_BIN_PATH=
#export DEFAULT_COMPILER="INTEL"
#export NO_ECHO_FLAGS=
#export REGULAR_EXPRESSIONS_FOR_NO_ECHO=
#export STRING_PREPEND_TO_ECHO=
#export DEBUG_LOG_FILE=tmp.log

#export CC=icc
#export CFLAGS=
#export LDFLAGS="-shared-intel $CFLAGS"
#export LIBS=
#export CPPFLAGS=
#export CPP="icc -E"
#export CCAS=
#export CCASFLAGS=
#export CXX=icpc
#export CXXFLAGS=
#export CXXCPP="icpc -E"
#export F77=ifort
#export FFLAGS="$CFLAGS"

alias die='_error_exit_ "Error in file $0 at line $LINENO\n"'

function special_rules()
{
    return
    local arg=
    for arg in $*; do
	echo $arg
    done
}

function main() 
{
    module purge
    export LD_LIBRARY_PATH=
    module load intel/14.0.2
    module load cuda/6.5.12
    
    local util=$HOME/bin/intel/util.sh
    if [ -e $util ]; then
	source $util
    fi
    
    export SPECIAL_RULES_FUNCTION=special_rules
    if [ "$SPECIAL_RULES_FUNCTION" != "" ]; then
	export BUILD_WRAPPER_SCRIPT=$(readlink -e $0)
    fi
    
    export GNU_BIN_PATH=$(dirname $(which gcc))
    export INTEL_BIN_PATH=$(dirname $(which icc))
    #export INTEL_MPI_BIN_PATH=$(dirname $(which mpicc))
    export NVCC_BIN_PATH=$(dirname $(which nvcc))
    
    export INVALID_FLAGS_FOR_GNU_COMPILERS="-O -O0 -O1 -O2 -g"
    export OPTIMIZATION_FLAGS_FOR_GNU_COMPILERS="-O3 -fPIC -fopenmp -msse4.2 -mno-avx"
    
    export INVALID_FLAGS_FOR_INTEL_COMPILERS="-O -O0 -O1 -O2 -g -lm -xhost -fast"
    export OPTIMIZATION_FLAGS_FOR_INTEL_COMPILERS="-O3 -fPIC -unroll -ip -axavx -xsse4.2 -openmp -vec-report -par-report -openmp-report -Wno-deprecated"
    export OPTIMIZATION_FLAGS_FOR_INTEL_FORTRAN_COMPILERS="-O3 -fPIC -unroll -ip -axavx -xsse4.2 -openmp -vec-report -par-report -openmp-report"
    
    export CPPFLAGS=$(for inc in $(env | grep _INC= | cut -d= -f2); do echo '-I'$inc; done | xargs)
    export LDFLAGS=$(for lib in $(env | grep _LIB= | cut -d= -f2); do echo '-L'$lib; done | xargs)
    
    prepend_to_env_variable INCLUDE_FLAGS "$CPPFLAGS"
    prepend_to_env_variable LINK_FLAGS "$LDFLAGS"
    
    export LINK_FLAGS_FOR_INTEL_COMPILERS="-shared-intel"
    export EXTRA_LINK_FLAGS="$(LD_LIBRARY_PATH_to_rpath)"

    #export INCLUDE_FLAGS=
    #export LINK_FLAGS=

    if [ "$DEBUG_LOG_FILE" != "" ]; then
	rm -rf $DEBUG_LOG_FILE
    fi
    
    export LD_RUN_PATH=$LD_LIBRARY_PATH
    
    local prefix=$(pwd)
    if [ "$prefix" == "" -a -d ../local ]; then
        prefix=$(readlink -e ../local)
    fi
    if [ "$prefix" == "" ]; then
        die "$0: no prefix defined"
    fi
    
    local args=$*
    local arg=
    for arg in $args; do
	
	case $arg in
	    
	    configure|conf)
		echo " Run configuration ..."
		#export PATH=.:$HOME/bin/intel:$PATH
		
		if [ "$DEFAULT_COMPILER" != "GNU" ]; then
		    export CC=icc
                    export CXX=icpc
                    export F77=ifort
                    export FC=ifort
		fi
		
		./configure --build=x86_64-centos-linux \
		    --prefix=$prefix
		;;
	    
	    cmake)
		module load cmake/intel/2.8.12.2
		export PATH=.:$HOME/bin/intel:$PATH
		
                export CC=icc
                export CXX=icpc
		cmake \
		    -DCMAKE_BUILD_TYPE=release \
		    -DBUILD_STATIC_LIBS::BOOL=ON \
                    -DBUILD_SHARED_LIBS::BOOL=ON \
                    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
                    -DCMAKE_SKIP_RPATH:BOOL=ON \
		    -DCMAKE_INSTALL_PREFIX:PATH=$prefix \
		    ../breakdancer
                ;;
	    
	    make)
		export PATH=.:$HOME/bin/intel:$PATH
		echo " Run make"
		eval "$args" 
		exit
		;;

	    a2so)
		export PATH=.:$HOME/bin/intel:$PATH
		cd $SUITESPARSE_LIB
		icc -shared -o libsuitesparse.so  \
		    -Wl,--whole-archive \
		    libamd.a \
		    -Wl,--no-whole-archive \
		    -L$MKL_ROOT/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lrt
		exit
		;;
	    
	    *)
		die " Usage: $0 <argument>: configure make"
		;;
	esac

	args=$(eval "echo $args | sed -e 's/$arg //'")
    done
}

## do the main work here
## do not modify the follwoing part, we just need to modify the main function

if [ "$TO_SOURCE_BUILD_WRAPPER_SCRIPT" == "" ]; then
    main "$*"
    exit
else
    unset -f main
fi

## do not add anything after this line