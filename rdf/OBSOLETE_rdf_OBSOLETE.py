from pymatgen.analysis.diffusion.aimd.rdf import RadialDistributionFunction, RadialDistributionFunctionFast
from pymatgen.core import Structure
import matplotlib.pyplot as plt

cif_file = "test_cif_files/Fe.cif"

try:
    structure_obj_1 = Structure.from_file(cif_file)
    structure_obj_2 = Structure.from_file("test_cif_files/Mg.cif")
    #struct_list = [structure_obj_1, structure_obj_2]
    struct_list_Fe = [structure_obj_1]
    struct_list_Mg = [structure_obj_2]

    rdf_calc = (RadialDistributionFunction(structures= struct_list_Mg, ngrid=200, sigma=0.2))
    rdf_calc.export_rdf(filename="Mg_rdf.csv")
    rdf_calc.get_rdf_plot()

    #rdf_fast_calc = RadialDistributionFunctionFast(struct_list).get_one_rdf(ref_species="Fe",species="Fe",index=0)

    plt.title("Average RDF for Two Structures")
    plt.show()



except FileNotFoundError:
    print(f"Could not find {cif_file}.")