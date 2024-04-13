#ifndef PARSER_H
#define PARSER_H

#include "defines.h"

class evolver;

class parser
{
    private:
        int populate_system();
        evolver *system;
        std::map<std::string, float> parameters;
        std::vector<char> token_split_char = {'+', '-', '='};
        std::vector<char> factor_split_char = {'*', '/'};
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

    public:
        int createFromFile(const std::string &);
        parser(evolver *system);
        int add_equation(const std::string &);
        int insert_parameter(const std::string &, float);
        int exists_parameter(const std::string &);
        int is_split_character(char);
        int get_token(const std::string &, int);
        int get_factor(const std::string &, int);
        int verbose;
};

#endif

