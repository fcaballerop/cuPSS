#ifndef PARSER_H
#define PARSER_H

#include "defines.h"

class evolver;

class parser {
    private:
        bool verbose;
        int populate_system();
        evolver *system;
        std::map<std::string, float> parameters;
        float is_numerical_factor(const std::string &);
        int is_q2n_factor(const std::string &);
        int is_iq_factor(const std::string &, const std::string &);
        int is_field(const std::string &);
        int is_number(const std::string &);
        int number_of_fields(const std::string &);
        std::string get_field_name(const std::string &);
        std::string extract_field_name(const std::string &);
        void get_factors(const std::string &, std::vector<std::string> &);
        void get_par_terms(const std::string &, std::vector<std::string> &);

        int split_equation_sides(std::string input, std::string &lhs, std::string &rhs);
        int remove_spaces(std::string &input);
        int split_terms(std::string input, std::vector<std::string> &terms);
        int expand_once(std::string input, std::vector<std::string> &terms);
        int expand_all(std::vector<std::string> &terms);
        int get_fields(std::string term, std::vector<std::string> &fields);
        int get_field_vector(const std::string &, std::vector<std::string> &fields);
        pres get_prefactor(const std::string &);

    public:
        int createFromFile(const std::string &);
        parser(evolver *system);
        int add_equation(const std::string &);
        pres add_noise(const std::string &);
        int insert_parameter(const std::string &, float);
        int exists_parameter(const std::string &);
        void writeParamsToFile(const std::string &);
        float getParameter(const std::string &);
        int changeParameter(const std::string &, float new_value);
        
        int isParameterInString(const std::string &, const std::string &);

        int recalculateImplicits(const std::vector<std::string> &, std::vector<pres> &, int);
};

#endif