/**rac-extractor.cpp
 *
 * Implements the RAC Extractor class
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <Eigen/Dense>
#include <jsoncpp/json/json.h>

struct Atom
{
};

class mol3D
{
public:
    void addAtom(Atom atom)
    {
    }
};

void writeXYZandGraph(std::string filename, const std::vector<std::string> &atom_labels, const mat &cell, const mat &fcoords_connected, const mat &adj_mat)
{
}

void append_descriptors(std::vector<std::string> &descriptor_names, std::vector<double> &descriptors, const std::vector<std::string> &colnames, const std::vector<double> &results, std::string prefix, std::string scope)
{
}

std::vector<double> generate_atomonly_autocorrelations(mol3D &mol, const std::vector<int> &atom_indices, bool loud, int depth, bool oct, bool polarizability)
{
}

std::vector<double> generate_atomonly_deltametrics(mol3D &mol, const std::vector<int> &atom_indices, bool loud, int depth, bool oct, bool polarizability)
{
}

std::vector<double> generate_full_complex_autocorrelations(mol3D &mol, int depth, bool loud, bool flag_name)
{
}

std::vector<double> generate_multimetal_autocorrelations(mol3D &mol, int depth, bool loud)
{
}

std::vector<double> generate_multimetal_deltametrics(mol3D &mol, int depth, bool loud)
{
}

tuple<std::vector<std::string>, std::vector<double>, std::vector<std::string>, std::vector<double>> make_MOF_SBU_RACs(const std::vector<std::vector<int>> &SBUlist, const std::vector<sp_mat> &SBU_subgraph, mol3D &molcif, int depth, const std::string &name, const mat &cell, const std::vector<int> &anchoring_atoms, bool sbupath = false, const std::vector<std::vector<int>> &connections_list = {}, const std::vector<sp_mat> &connections_subgraphlist = {})
{
    std::vector<std::vector<double>> descriptor_list;
    std::vector<std::vector<double>> lc_descriptor_list;
    std::vector<std::string> lc_names;
    std::vector<std::string> names;
    int n_sbu = SBUlist.size();
    std::vector<std::string> descriptor_names;
    std::vector<double> descriptors;
    Json::Value sbu_descriptors;
    Json::Value lc_descriptors;

    if (sbupath)
    {
        std::string sbu_descriptor_path = fs::path(sbupath).parent_path().std::string();
        if (fs::file_size(sbu_descriptor_path + "/sbu_descriptors.csv") > 0)
        {
            std::std::ifstream sbu_file(sbu_descriptor_path + "/sbu_descriptors.csv");
            sbu_file >> sbu_descriptors;
            sbu_file.close();
        }

        if (fs::file_size(sbu_descriptor_path + "/lc_descriptors.csv") > 0)
        {
            std::ifstream lc_file(sbu_descriptor_path + "/lc_descriptors.csv");
            lc_file >> lc_descriptors;
            lc_file.close();
        }
    }

    for (int i = 0; i < SBUlist.size(); ++i)
    {
        descriptor_names.clear();
        descriptors.clear();
        mol3D SBU_mol;

        for (const auto &val : SBUlist[i])
        {
            SBU_mol.addAtom(molcif.atoms[val]);
        }
        SBU_mol.graph = conv_to<mat>::from(SBU_subgraph[i]);

        for (int j = 0; j < connections_list.size(); ++j)
        {
            descriptor_names.clear();
            descriptors.clear();
            if (set<int>(SBUlist[i].begin(), SBUlist[i].end()).count(connections_list[j]))
            {
                mol3D temp_mol;
                std::vector<int> link_list;

                for (int k = 0; k < connections_list[j].size(); ++k)
                {
                    if (find(anchoring_atoms.begin(), anchoring_atoms.end(), connections_list[j][k]) != anchoring_atoms.end())
                    {
                        link_list.push_back(k);
                    }
                    temp_mol.addAtom(molcif.atoms[connections_list[j][k]]);
                }

                temp_mol.graph = conv_to<mat>::from(connections_subgraphlist[j]);
                auto results_dictionary = generate_atomonly_autocorrelations(temp_mol, link_list, false, depth, false, true);
                append_descriptors(descriptor_names, descriptors, results_dictionary, results_dictionary, "lc", "all");

                results_dictionary = generate_atomonly_deltametrics(temp_mol, link_list, false, depth, false, true);
                append_descriptors(descriptor_names, descriptors, results_dictionary, results_dictionary, "D_lc", "all");

                std::vector<int> functional_atoms;
                for (int k = 0; k < temp_mol.graph.n_rows; ++k)
                {
                    if (find(link_list.begin(), link_list.end(), k) == link_list.end() && molcif.atoms[k].sym != "C" && molcif.atoms[k].sym != "H")
                    {
                        functional_atoms.push_back(k);
                    }
                }

                if (!functional_atoms.empty())
                {
                    results_dictionary = generate_atomonly_autocorrelations(temp_mol, functional_atoms, false, depth, false, true);
                    append_descriptors(descriptor_names, descriptors, results_dictionary, results_dictionary, "func", "all");

                    results_dictionary = generate_atomonly_deltametrics(temp_mol, functional_atoms, false, depth, false, true);
                    append_descriptors(descriptor_names, descriptors, results_dictionary, results_dictionary, "D_func", "all");
                }
                else
                {
                    append_descriptors(descriptor_names, descriptors, results_dictionary, std::vector<double>(6 * (depth + 1), 0.0), "func", "all");
                    append_descriptors(descriptor_names, descriptors, results_dictionary, std::vector<double>(6 * (depth + 1), 0.0), "D_func", "all");
                }

                for (const auto &val : descriptors)
                {
                    if (typeid(val) != typeid(double))
                    {
                        throw runtime_error("Mixed typing. Please convert to C++ double.");
                    }
                }

                descriptor_names.push_back("name");
                descriptors.push_back(name);
                Json::Value desc_dict;
                for (int k = 0; k < descriptor_names.size(); ++k)
                {
                    desc_dict[descriptor_names[k]] = descriptors[k];
                }
                descriptors.pop_back();
                descriptor_names.pop_back();
                lc_descriptors.append(desc_dict);
                lc_descriptor_list.push_back(descriptors);

                if (j == 0)
                {
                    lc_names = descriptor_names;
                }
            }
        }

        auto averaged_lc_descriptors = mean(mat(lc_descriptor_list));
        ofstream lc_file(sbu_descriptor_path + "/lc_descriptors.csv");
        lc_file << lc_descriptors;
        lc_file.close();

        descriptors.clear();
        descriptor_names.clear();
        mat SBU_mol_cart_coords(SBU_mol.atoms.size(), 3);
        std::vector<std::string> SBU_mol_atom_labels(SBU_mol.atoms.size());
        mat SBU_mol_adj_mat = SBU_mol.graph;

        for (int j = 0; j < SBU_mol.atoms.size(); ++j)
        {
            SBU_mol_cart_coords.row(j) = SBU_mol.atoms[j].coords.t();
            SBU_mol_atom_labels[j] = SBU_mol.atoms[j].sym;
        }

        if (sbupath && !fs::exists(sbupath + "/" + name + to_std::string(i) + ".xyz"))
        {
            std::string xyzname = sbupath + "/" + name + "_sbu_" + to_std::string(i) + ".xyz";
            auto SBU_mol_fcoords_connected = XYZ_connected(cell, SBU_mol_cart_coords, SBU_mol_adj_mat);
            writeXYZandGraph(xyzname, SBU_mol_atom_labels, cell, SBU_mol_fcoords_connected, SBU_mol_adj_mat);
        }

        auto results_dictionary = generate_full_complex_autocorrelations(SBU_mol, depth, false, false);
        append_descriptors(descriptor_names, descriptors, results_dictionary, results_dictionary, "f", "all");

        results_dictionary = generate_multimetal_autocorrelations(molcif, depth, false);
        append_descriptors(descriptor_names, descriptors, results_dictionary, results_dictionary, "mc", "all");

        results_dictionary = generate_multimetal_deltametrics(molcif, depth, false);
        append_descriptors(descriptor_names, descriptors, results_dictionary, results_dictionary, "D_mc", "all");

        descriptor_names.push_back("name");
        descriptors.push_back(name);
        Json::Value desc_dict;
        for (int j = 0; j < descriptor_names.size(); ++j)
        {
            desc_dict[descriptor_names[j]] = descriptors[j];
        }
        descriptors.pop_back();
        descriptor_names.pop_back();
        sbu_descriptors.append(desc_dict);
        descriptor_list.push_back(descriptors);

        if (i == 0)
        {
            names = descriptor_names;
        }
    }

    std::ofstream sbu_file(sbu_descriptor_path + "/sbu_descriptors.csv");
    sbu_file << sbu_descriptors;
    sbu_file.close();

    auto averaged_SBU_descriptors = mean(mat(descriptor_list));

    return make_tuple(names, averaged_SBU_descriptors, lc_names, averaged_lc_descriptors);
}
