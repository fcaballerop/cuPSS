#include "../inc/parser.h"
#include "../inc/evolver.h"
#include "../inc/field.h"
#include <sstream>
#include <fstream>


parser::parser(evolver *_system)
{
    system = _system;
    verbose = 0;
}

int parser::createFromFile(const std::string &_file_name)
{
    std::string line;
    std::cout << "Trying to read system from " << _file_name << std::endl;
    std::ifstream infile(_file_name);
    int read_type = -1; // 0 fields, 1 parameters, 2 equations
    std::vector<std::string> equation_vector;
    while (std::getline(infile, line))
    {
        std::cout << line << std::endl;
        std::istringstream iss(line);
        if (line == "" || line.substr(0,0) == "#")
            continue;
        if (read_type == -1)
        {
            // must read a type
            if (line.substr(0,6) == "Fields" || line.substr(0,6) == "fields")
                read_type = 0;
            else if (line.substr(0,6) == "Parameters" || line.substr(0,6) == "parameters")
                read_type = 1;
            else if (line.substr(0,6) == "Equations" || line.substr(0,6) == "Equations")
                read_type = 2;
            else
            {
                std::cout << "ERROR: reading file " << _file_name << std::endl;
                std::cout << "First line must be either fields, parameters, or equations, it is:" << std::endl;
                std::cout << line << std::endl;
                return -1;
            }
            continue;
        }
        if (line.substr(0,6) == "Fields" || line.substr(0,6) == "fields")
        {
            read_type = 0;
            continue;
        }
        else if (line.substr(0,10) == "Parameters" || line.substr(0,10) == "parameters")
        {
            read_type = 1;
            continue;
        }
        else if (line.substr(0,9) == "Equations" || line.substr(0,9) == "Equations")
        {
            read_type = 2;
            continue;
        }
        else
        {
            if (read_type == 0)
            {
                std::string field_name;
                int dynamic;
                int output;
                if (!(iss >> field_name >> dynamic >> output))
                {
                    std::cout << "Error reading field line: " << line << std::endl;
                    std::cout << "Must be: field_name dynamic_value output" << std::endl;
                    return -1;
                }
                std::cout << "Creating field: " << field_name << ", dynamic: " << dynamic << std::endl;
                system->createField(field_name, dynamic);
                system->setOutputField(field_name, output);
            }
            if (read_type == 1)
            {
                std::string param_name;
                float value;
                if (!(iss >> param_name >> value))
                {
                    std::cout << "Error reading parameter line: " << line << std::endl;
                    std::cout << "Must be: param_name value" << std::endl;
                    return -1;
                }
                std::cout << "Creating parameter: " << param_name << " = " << value << std::endl;
                insert_parameter(param_name, value);
            }
            if (read_type == 2)
            {
                std::string param_name;
                equation_vector.push_back(line);
            }
        }
    }
    for (int i = 0; i < equation_vector.size(); i++)
    {
        add_equation(equation_vector[i]);
    }
    return 0;
}

int parser::insert_parameter(const std::string & p_name, float value)
{
    if (exists_parameter(p_name))
    {
        std::cout << "Duplicate parameter " << p_name << std::endl;
        return -1;
    }
    parameters[p_name] = value;
    return 0;
}

int parser::exists_parameter(const std::string &p_name)
{
    std::map<std::string, float>::iterator it = parameters.begin();

    while (it != parameters.end())
    {
        if (it->first == p_name)
        {
            return 1;
        }
        it++;
    }

    return 0;
}

int parser::is_split_character(char character)
{
    for (int i = 0; i < token_split_char.size(); i++)
    {
        if (character == token_split_char[i])
        {
            return 1;
        }
    }
    return 0;
}

int parser::get_token(const std::string &line, int start)
{
    if (!is_split_character(line[start]))
    {
        std::cout << "get_token() error, first character is not split character: " << line[start] << std::endl;
        return -1;
    }
    int endf = start + 1;
    while (!is_split_character(line[endf]) && endf != line.size()) endf += 1;
    return endf;
}

int parser::get_factor(const std::string &line, int start)
{
    int endf = start + 1;
    while (line[endf] != '*' && endf != line.size()) endf += 1;
    return endf;
}

int parser::is_number(const std::string &term)
{
    int _is_number = 1;
    int has_period = 0;
    int period_ind = 0;
    std::string to_test = term;
    if (term.substr(0,2) == "1/")
        to_test.erase(0,2);
    for (int i = 0; i < to_test.size(); i++)
    {
        if (to_test[i] == '.')
        {
            has_period++;
            period_ind = i;
        }
    }
    if (has_period > 1)
    {
        std::cout << "ERROR: prefactor not a parameter and not a number: " << term << std::endl;
    }
    if (has_period)
        to_test.erase(period_ind,1);
    for (int i = 0; i < to_test.size(); i++)
    {
        char this_char = to_test.at(i);
        if (!isdigit(this_char))
            _is_number = 0;
    }
    return _is_number;
}

float parser::is_numerical_factor(const std::string &factor)
{
    if (is_q2n_factor(factor)) return 1.0f;
    if (is_iq_factor(factor, "iqx")) return 1.0f;
    if (is_iq_factor(factor, "iqy")) return 1.0f;
    if (is_iq_factor(factor, "1/q")) return 1.0f;
    if (is_field(factor)) return 1.0f;

    int divides = 0;
    std::string _factor;
    if (factor.substr(0,2) == "1/")
    {
        divides = 1;
        _factor = factor.substr(2);
    }
    else
        _factor = factor;


    if (exists_parameter(_factor))
    {
        if (divides)
            return 1.0f/parameters[_factor];
        else
            return parameters[_factor];
    }
    else if (is_number(_factor))
    {
        if (divides)
            return 1.0f/std::stof(_factor);
        else
            return std::stof(_factor);
    }
    else
    {
        std::cout << "ERROR, parameter not found and not a number: " << factor << std::endl;
    }
    return 1.0f;
}

int parser::is_field(const std::string &factor)
{
    int power = 1;
    int ppos = 0;
    for (int i = 0; i < factor.size(); i++)
    {
        if (factor[i] == '^')
        {
            power = atoi(factor.substr(i+1).c_str());
            ppos = i;
        }
    }
    if (power != 1)
    {
        std::string field_name = factor.substr(0, ppos);
        for (int i = 0; i < system->fields.size(); i++)
        {
            if (field_name == system->fields[i]->name) return power;
        }
    }
    else
    {
        for (int i = 0; i < system->fields.size(); i++)
        {
            if (factor == system->fields[i]->name) return power;
        }
    }
    return 0;
}

int parser::is_q2n_factor(const std::string &factor)
{
    if (factor.size() < 3) return 0;

    if (factor[0] == 'q' && factor[1] == '^')
    {
        std::string number_str = factor.substr(2);
        return atoi(number_str.c_str());
    }

    return 0;
}

int parser::is_iq_factor(const std::string &factor, const std::string &iq)
{
    if (factor.size() < 3) return 0;
    if (factor.substr(0,3) == iq && factor.size() == 3) return 1;
    // if (factor.substr(0,3) == iq && factor[3] != '^')
    // {
    //     std::cout << "ERROR with factor " << factor << std::endl;
    //     return -1;
    // }
    else if (factor.substr(0,3) == iq)
    {
        std::string number_str = factor.substr(4);
        return atoi(number_str.c_str());
    }
    return 0;
}

std::string parser::extract_field_name(const std::string &_term)
{
    std::string _name;
    if (is_field(_term) == 1)
        return _term;
    else
    {
        int chevron = 0;
        for (int i = 0; i < _term.size(); i++)
        {
            if (_term[i] == '^')
                chevron = i;
        }
        return _term.substr(0, chevron);
    }
}
std::string parser::get_field_name(const std::string &_term)
{
    std::vector<std::string> factors;
    std::string field_name;
    int start = 0;
    int pointer = 0;
    int open_par = 0;
    while (pointer < _term.size())
    {
        if (_term[pointer] == '(')
            open_par = 1;
        if (_term[pointer] == ')')
            open_par = 0;
        if (_term[pointer] == '*' && open_par == 0)
        {
            factors.push_back(_term.substr(start, pointer-start));
            start = pointer+1;
        }
        if (pointer == _term.size() - 1)
        {
            factors.push_back(_term.substr(start));
        }
        pointer++;
    }
    for (int i = 0; i < factors.size(); i++)
    {
        if (is_field(factors[i]) > 0)
            field_name = factors[i];
    }
    return field_name;
}

int parser::number_of_fields(const std::string &_term)
{
    std::vector<std::string> factors;
    int field_counter = 0;
    int start = 0;
    int pointer = 0;
    int open_par = 0;
    while (pointer < _term.size())
    {
        if (_term[pointer] == '(')
            open_par = 1;
        if (_term[pointer] == ')')
            open_par = 0;
        if (_term[pointer] == '*' && open_par == 0)
        {
            factors.push_back(_term.substr(start, pointer-start));
            start = pointer+1;
        }
        if (pointer == _term.size() - 1)
        {
            factors.push_back(_term.substr(start));
        }
        pointer++;
    }
    for (int i = 0; i < factors.size(); i++)
    {
        field_counter += is_field(factors[i]);
    }
    return field_counter;
}

void parser::get_factors(const std::string &_term, std::vector<std::string> &factors)
{
    int start = 0;
    int pointer = 0;
    int open_par = 0;
    while (pointer < _term.size())
    {
        if (_term[pointer] == '(')
            open_par = 1;
        if (_term[pointer] == ')')
            open_par = 0;
        if (_term[pointer] == '*' && open_par == 0)
        {
            factors.push_back(_term.substr(start, pointer-start));
            start = pointer+1;
        }
        if (pointer == _term.size() - 1)
        {
            factors.push_back(_term.substr(start));
        }
        pointer++;
    }
}

void parser::get_par_terms(const std::string &_term, std::vector<std::string> &factors)
{
    if (!(_term[0] == '(' && _term[_term.size()-1] == ')'))
    {
        std::cout << "ERROR: get_par_terms not between parenthesis" << std::endl;
        std::cout << _term << std::endl;
        return;
    }
    std::string _term_np = _term.substr(1, _term.size()-2);

    int start = 0;
    int pointer = 1;

    while (pointer < _term_np.size())
    {
        if (is_split_character(_term_np[pointer]))
        {
            factors.push_back(_term_np.substr(start, pointer-start));
            start = pointer;
        }
        pointer++;
    }
    // last term 
    factors.push_back(_term_np.substr(start));
    return;
}

int parser::add_equation(const std::string &_equation)
{
    std::string equation = _equation;
    std::cout << "Parsing equation:" << std::endl << equation << std::endl;  
    for (int i = 0; i < equation.size(); i++)
    {
        if (equation[i] == ' ')
        {
            equation.erase(i,1);
            i--;
        }
    }
    if (verbose)
        std::cout << "Without spaces:" << std::endl << equation << std::endl;  

    int dynamic = 0;
    std::string field_name;
    std::vector<std::string> lhs_terms;
    std::vector<std::string> rhs_terms;
    int pointer=0;
    int start = 0;
    int open_parenthesis = 0;
    int reached_equal = 0;
    int field_index = -1;

    while (pointer < equation.size())
    {
        // Add lhs terms
        if ((is_split_character(equation[pointer]) && open_parenthesis == 0))
        {
            if (reached_equal)
                rhs_terms.push_back(equation.substr(start, pointer-start));
            else
            {
                lhs_terms.push_back(equation.substr(start, pointer-start));
                if (equation[pointer] == '=')
                {
                    reached_equal = 1;
                    pointer++;
                    if (pointer >= equation.size()) break;
                }
            }
            start = pointer;
        }
        if (pointer == equation.size()-1)
        {
            if (reached_equal)
                rhs_terms.push_back(equation.substr(start));
            else
                lhs_terms.push_back(equation.substr(start));
            break;
        }
        if ((equation[pointer] == '(' && open_parenthesis == 1) ||
                (equation[pointer] == ')' && open_parenthesis == 0))
        {
            std::cout << "ERROR: Parenthesis error processing equation" << std::endl;
            std::cout << equation << std::endl;
            for (int i = 0; i < pointer;i++) std::cout << " ";
            std::cout << "^" << std::endl;
            return -1;
        }
        if ((equation[pointer] == '(' && open_parenthesis == 0) ||
                (equation[pointer] == ')' && open_parenthesis == 1))
        {
            open_parenthesis = 1 - open_parenthesis;
        }
        pointer++;
    }

    // Find which field we're creating an equation for
    if (lhs_terms[0].substr(0,2) == "dt")
    {
        dynamic = 1;
        field_name = lhs_terms[0].substr(2);
        lhs_terms.erase(lhs_terms.begin());
    }
    else
    {
        if (number_of_fields(lhs_terms[0]) != 1)
        {
            std::cout << "ERROR: Nonlinear term in left hand side" << std::endl;
            std::cout << lhs_terms[0] << std::endl;
            return -1;
        }
        field_name = get_field_name(lhs_terms[0]);
    }

    for (int i = 0; i < system->fields.size(); i++)
    {
        if (system->fields[i]->name == field_name)
            field_index = i;
    }
    if (field_index < 0)
    {
        std::cout << "ERROR: field " << field_name << " not found in evolver" << std::endl;
        return -1;
    }

    if (verbose)
    {
        std::cout << "Processing terms for: " << field_name << " " << dynamic << std::endl;
        std::cout << "Left hand side terms:" << std::endl;
        for (int i = 0; i < lhs_terms.size(); i++)
        {
            std::cout << lhs_terms[i] << std::endl;
        }
        std::cout << "Right hand side terms:" << std::endl;
        for (int i = 0; i < rhs_terms.size(); i++)
        {
            std::cout << rhs_terms[i] << std::endl;
        }
    }

    // Process left hand side
    for (int i = 0; i < lhs_terms.size(); i++)
    {
        int negative = 0;
        if (lhs_terms[i][0] == '-')
        {
            negative = 1;
            lhs_terms[i].erase(lhs_terms[i].begin());
        }
        if (lhs_terms[i][0] == '+')
        {
            lhs_terms[i].erase(lhs_terms[i].begin());
        }
        std::vector<std::string> factors;
        get_factors(lhs_terms[i], factors);
        if (verbose)
        {
            std::cout << "Processing lhs term: " << lhs_terms[i] << std::endl;
            for (int j = 0; j < factors.size(); j++)
            {
                std::cout << factors[j] << std::endl;
            }
        }

        if (lhs_terms[i][0] == '(' || lhs_terms[i][lhs_terms[i].size()-1] == ')')
        {
            int par_index = 0;
            if (lhs_terms[i][lhs_terms[i].size()-1] == ')')
                par_index = 1;
            std::vector<std::string> par_terms;
            get_par_terms(factors[par_index], par_terms);
            if (verbose)
            {
                std::cout << "Par_terms for " << lhs_terms[i] << std::endl;
                for (int j = 0; j < par_terms.size(); j++)
                    std::cout << par_terms[j] << std::endl;
            }
            for (int j = 0; j < par_terms.size(); j++)
            {
                pres this_prefactor = {1.0f, 0, 0, 0, 0};
                if (par_terms[j][0] == '+') par_terms[j].erase(0,1);
                if (par_terms[j][0] == '-')
                {
                    par_terms[j].erase(0,1);
                    this_prefactor.preFactor *= -1.0f;
                }
                std::vector<std::string> par_factors;
                get_factors(par_terms[j], par_factors);
                for (int k = 0; k < par_factors.size(); k++)
                {
                    this_prefactor.preFactor *= is_numerical_factor(par_factors[k]);
                    this_prefactor.q2n += is_q2n_factor(par_factors[k])/2;
                    this_prefactor.iqx += is_iq_factor(par_factors[k], "iqx");
                    this_prefactor.iqy += is_iq_factor(par_factors[k], "iqy");
                    this_prefactor.invq += is_iq_factor(par_factors[k], "1/q");
                }
                if (negative && !dynamic)
                    this_prefactor.preFactor *= -1.0f;
                if (!negative && dynamic)
                    this_prefactor.preFactor *= -1.0f;
                system->fields[field_index]->implicit.push_back(this_prefactor);
            }
        }
        else // just one factor
        {
            pres this_prefactor = {1.0f, 0, 0, 0, 0};
            for (int j = 0; j < factors.size(); j++)
            {
                this_prefactor.preFactor *= is_numerical_factor(factors[j]);
                this_prefactor.q2n += is_q2n_factor(factors[j])/2;
                this_prefactor.iqx += is_iq_factor(factors[j], "iqx");
                this_prefactor.iqy += is_iq_factor(factors[j], "iqy");
                this_prefactor.invq += is_iq_factor(factors[j], "1/q");
            }
            if (negative && !dynamic)
                this_prefactor.preFactor *= -1.0f;
            if (!negative && dynamic)
                this_prefactor.preFactor *= -1.0f;
            system->fields[field_index]->implicit.push_back(this_prefactor);
        }
    }


    // Process rhs terms

    for (int i = 0; i < rhs_terms.size(); i++)
    {
        if (verbose)
        {
            std::cout << "Processing rhs term: " << rhs_terms[i] << std::endl;
        }
        int negative = 0;
        std::vector<pres> prefactors;
        std::vector<std::string> field_strings;
        if (rhs_terms[i][0] == '+') rhs_terms[i].erase(0,1);
        if (rhs_terms[i][0] == '-') 
        {
            rhs_terms[i].erase(0,1);
            negative = 1;
        }
        std::vector<std::string> factors;
        get_factors(rhs_terms[i], factors);
        if (verbose)
        {
            std::cout << "Factors:" << std::endl;
            for (int j = 0; j < factors.size(); j++)
                std::cout << factors[j] << std::endl;
        }
        // Each factor must be either a field, a field power, or a prefactor
        int is_one_factor = 0;
        pres full_prefactor = {1.0f, 0, 0, 0, 0};
        for (int f = 0; f < factors.size(); f++)
        {
            if (is_field(factors[f]))
            {
                std::string this_field;
                this_field = extract_field_name(factors[f]);
                if (verbose)
                {
                    std::cout << "Factor " << factors[f] << " is field: " << this_field << std::endl;
                }
                for (int p = 0; p < is_field(factors[f]); p++)
                {
                    field_strings.push_back(this_field);
                }
            }
            else
            {
                if (factors[f][0] == '(')
                {
                    std::vector<std::string> par_terms;
                    get_par_terms(factors[f], par_terms);
                    for (int k = 0; k < par_terms.size(); k++)
                    {
                        pres _this_p = {1.0f, 0, 0, 0, 0};
                        if (par_terms[k][0] == '+') par_terms[k].erase(0,1);
                        if (par_terms[k][0] == '-')
                        {
                            par_terms[k].erase(0,1);
                            _this_p.preFactor = -1.0f;
                        }
                        std::vector<std::string> factors_this_p;
                        get_factors(par_terms[k], factors_this_p);
                        for (int fp = 0; fp < factors_this_p.size(); fp++)
                        {
                            _this_p.preFactor *= is_numerical_factor(factors_this_p[fp]);
                            _this_p.q2n += is_q2n_factor(factors_this_p[fp])/2;
                            _this_p.iqx += is_iq_factor(factors_this_p[fp], "iqx");
                            _this_p.iqy += is_iq_factor(factors_this_p[fp], "iqy");
                            _this_p.invq += is_iq_factor(factors_this_p[fp], "1/q");
                        }
                        prefactors.push_back(_this_p);
                    }
                }
                else
                {
                    is_one_factor = 1;
                    full_prefactor.preFactor *= is_numerical_factor(factors[f]);
                    full_prefactor.q2n += is_q2n_factor(factors[f])/2;
                    full_prefactor.iqx += is_iq_factor(factors[f], "iqx");
                    full_prefactor.iqy += is_iq_factor(factors[f], "iqy");
                    full_prefactor.invq += is_iq_factor(factors[f], "1/q");
                }
            }
        }
        if (is_one_factor) 
            prefactors.push_back(full_prefactor);
        if (prefactors.size() == 0)
            prefactors.push_back({1.0f, 0, 0, 0, 0});
        if (negative)
        {
            for (int p = 0; p < prefactors.size(); p++)
            {
                prefactors[p].preFactor *= -1.0f;
            }
        }
        system->createTerm(field_name, prefactors, field_strings);
    }

    std::cout << "DONE" << std::endl;
    return 0;
}
