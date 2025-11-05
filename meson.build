project('mindist', 'c', default_options : ['warning_level=2'],
)

add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

fc=meson.get_compiler('fortran')


# mindist_source = custom_target('utils',
#   input : ['lib/stdlib_sorting_sort_index_real.f90',
#            'lib/mindist_grid_utils.f90',
#            'lib/mindist_PBC_utils.f90',
#            'lib/mindist_PBC_grid_utils.f90'],
#   output : ['stdlib_sorting_sort_index_real.o',
#             'mindist_grid_utils.o',
#             'mindist_PBC_utils.o',
#             'mindist_PBC_grid_utils.o'],
#   command : [fc.cmd_array(), '-c', '-Wall', '-fPIC', '-Ofast', '@INPUT@']
# )

modules = ['mindist', 'mindist_grid', 'mindist_PBC', 'mindist_PBC_grid']
mod_sources = {}
foreach mod : modules
    if (mod=='mindist')
        wrp='-f2pywrappers.f'
    else
        wrp='-f2pywrappers2.f90'
    endif
    tgt = custom_target(
        mod+'module.c',
        input : ['lib/'+mod+'.f90'],
        output : [mod+'module.c', mod+wrp],
        command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', mod, '--lower']
    )
    mod_sources += {mod : tgt}
endforeach

py.install_sources(['src/__init__.py',
                    'src/mindist.py'],
                    subdir : 'mindist')

py.install_sources(['lib/__init__.py'],
                    subdir : 'mindist/_f2py')

foreach mod : modules
    files = ['lib/'+mod+'.f90']
    if (mod.contains('grid'))
      files += ['lib/stdlib_sorting_sort_index_real.f90',
                'lib/'+mod+'_utils.f90']
    endif
    if (mod.contains('PBC'))
      files += ['lib/mindist_PBC_utils.f90']
    endif
    py.extension_module(mod,
        [files, mod_sources[mod]],
        incdir_f2py / 'fortranobject.c',
        include_directories: inc_np,
        dependencies : py_dep,
        install : true,
        subdir : 'mindist/_f2py',
        link_args : ['-lgomp'],
        fortran_args : ['-fopenmp']
    )
endforeach