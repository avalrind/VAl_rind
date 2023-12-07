import setuptools

setuptools.setup(
    name = 'VAlrind' , 
    version = '0.0.1' , 
    author = 'AVal' , 
    description = 'Python Library for Training Nerual Networks' , 
    long_description = readme , 
    long_description_content_type = 'text/markdown' , 
    packages = setuptools.find_packages() , 
    classifiers = [
        'Programing Language :: Python :: 3' , 
        'License :: OSI Approved :: MIT License' , 
        'Operating System :: OS Independent' 
    ] , 
    python_requires = '>= 3.6' , 
    py_modules = ['val_rind'] , 
    package_dir = {'' : 'VAlrind/src'} , 
    install_requires = ['numpy']
)
