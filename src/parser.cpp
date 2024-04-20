#include "../inc/parser.h"
#include "../inc/evolver.h"
#include "../inc/field.h"
#include <sstream>
#include <fstream>
#include <algorithm>


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

// Returns power of the field, or 0 if it is not a field
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
    if (_term[0] == '+' || _term[0] == '-')
    {
        start = 1; pointer = 1;
    }
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
    std::string term = _term;
    if (term[0] == '+' || term[0] == '-')
        term = term.substr(1);
    std::vector<std::string> factors;
    int field_counter = 0;
    int start = 0;
    int pointer = 0;
    int open_par = 0;
    while (pointer < term.size())
    {
        if (term[pointer] == '(')
            open_par = 1;
        if (term[pointer] == ')')
            open_par = 0;
        if (term[pointer] == '*' && open_par == 0)
        {
            factors.push_back(term.substr(start, pointer-start));
            start = pointer+1;
        }
        if (pointer == term.size() - 1)
        {
            factors.push_back(term.substr(start));
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

pres parser::add_noise(const std::string &_equation)
{
    std::string equation = _equation;
    for (int i = 0; i < equation.size(); i++)
    {
        if (equation[i] == ' ')
        {
            equation.erase(i,1);
            i--;
        }
    }

    std::vector<std::string> factors;
    get_factors(equation, factors);

    pres prefactor = {1.0f, 0, 0, 0, 0};

    for (int i = 0; i < factors.size(); i++)
    {
        prefactor.preFactor *= is_numerical_factor(factors[i]);
        prefactor.q2n += is_q2n_factor(factors[i])/2;
        prefactor.invq += is_iq_factor(factors[i], "1/q");
    }

    return prefactor;
}

int parser::split_equation_sides(std::string input, std::string &lhs, std::string &rhs)
{
    int number_equals = 0;
    int equal_pos = 0;
    for (int i = 0; i < input.size(); i++)
    {
        if (input[i] == '=')
        {
            number_equals++;
            equal_pos = i;
        }
    }
    if (number_equals != 1)
    {
        std::cout << "Error processing" << std::endl;
        std::cout << input << std::endl;
        std::cout << "More than one equal sign" << std::endl;
        return -1;
    }
    lhs = input.substr(0,equal_pos);
    if (equal_pos == input.size()-1)
        rhs = "";
    else
        rhs = input.substr(equal_pos + 1);
    return 0;
}

int parser::remove_spaces(std::string &input)
{
    for (int i = 0; i < input.size(); i++)
    {
        if (input[i] == ' ')
        {
            if (i == input.size()-1)
                input = input.substr(0,i);
            else
                input = input.substr(0,i) + input.substr(i+1);
            i--;
        }
    }
    return 0;
}

int parser::split_terms(std::string input, std::vector<std::string> &terms)
{
    int current = 0;
    int pointer = 0;

    int par_level = 0;

    // std::cout << "Splitting " << input << std::endl;
    while (pointer < input.size())
    {
        if (input[pointer] == '(')
            par_level++;
        if (input[pointer] == ')')
            par_level--;
        if (pointer > 0 && par_level == 0 && (input[pointer] == '+' || input[pointer] == '-'))
        {
            // std::cout << "splitted " << input.substr(current, pointer-current) << std::endl;
            terms.push_back(input.substr(current, pointer-current));
            current = pointer;
        }
        pointer++;
    }
    // std::cout << "splitted " << input.substr(current) << std::endl;
    terms.push_back(input.substr(current));
    return 0;   
}

// will expand one set of parenthesis once
int parser::expand_once(std::string input, std::vector<std::string> &terms)
{
    int pointer = 0;
    int current = 0;

    int par_found = 0;
    int par_level = 0;

    while (pointer < input.size())
    {
        if (input[pointer] == '(')
        {
            if (pointer > 0 && input[pointer-1] == '/')
            {
                std::cout << "No dividing over parenthesis!" << std::endl;
                return -1;
            }
            par_found = 1;
            par_level++;
            if (par_level==1)
                current = pointer;
        }
        if (input[pointer] == ')')
            par_level--;
        if (par_level == 0 && par_found == 1)
        {
            std::vector<std::string> par_terms;
            split_terms(input.substr(current+1, pointer-current-1), par_terms);
            std::string pre = input.substr(0,current);
            std::string post = input.substr(pointer + 1);

            for (int k = 0; k < par_terms.size(); k++)
            {
                std::string this_pre = pre;
                if (par_terms[k].substr(0,1) == "+")
                    par_terms[k] = par_terms[k].substr(1);
                if (par_terms[k].substr(0,1) == "-")
                {
                    if (this_pre.substr(0,1) == "-")
                        this_pre = "+" + this_pre.substr(1);
                    else if (this_pre.substr(0,1) == "+")
                        this_pre = "-" + pre.substr(1);
                    else
                        this_pre = "-"+this_pre;
                    par_terms[k] = par_terms[k].substr(1);
                }
                terms.push_back(this_pre + par_terms[k] + post);
            }
            return 1;
        }
        pointer++;
    }
    return 0;
}

int parser::expand_all(std::vector<std::string> &terms)
{
    for (int k = 0; k < terms.size(); k++)
    {
        std::vector<std::string> sub_terms;
        if (expand_once(terms[k], sub_terms) > 0)
        {
            terms.erase(terms.begin()+k);
            terms.insert(terms.end(), sub_terms.begin(), sub_terms.end());
            k--;
        }
    }
    return 0;
}

pres parser::get_prefactor(std::string _term)
{
    std::string term = _term;
    pres prefactor = {1.0f, 0, 0, 0, 0};
    if (term[0] == '+')
    {
        term = term.substr(1);
    }
    if (term[0] == '-')
    {
        prefactor.preFactor = -1.0f;
        term = term.substr(1);
    }

    std::vector<std::string> factors;
    get_factors(term, factors);

    for (int i = 0; i < factors.size(); i++)
    {
        if (is_field(factors[i]) > 0)
            continue;
        else
        {
            prefactor.preFactor *= is_numerical_factor(factors[i]);
            prefactor.q2n += is_q2n_factor(factors[i])/2;
            prefactor.iqx += is_iq_factor(factors[i], "iqx");
            prefactor.iqy += is_iq_factor(factors[i], "iqy");
            prefactor.invq += is_iq_factor(factors[i], "1/q");
        }
    }

    return prefactor;
}

int parser::get_field_vector(std::string term, std::vector<std::string> &fields)
{
    if (term[0] == '+')
    {
        term = term.substr(1);
    }
    if (term[0] == '-')
    {
        term = term.substr(1);
    }
    std::vector<std::string> factors;
    get_factors(term, factors);

    for (int i = 0; i < factors.size(); i++)
    {
        if (is_field(factors[i]) > 0)
        {
            std::string this_field = extract_field_name(factors[i]);
            for (int p = 0; p < is_field(factors[i]); p++)
            {
                fields.push_back(this_field);
            }
        }
    }
    return 0;
}

int parser::add_equation(const std::string &_equation)
{
    std::string equation = _equation;
    remove_spaces(equation);
    
    std::cout << "Processing\n" << equation << "\n";
    
    int dynamic = 0;
    std::string field_name;
    std::string lhs, rhs;
    std::vector<std::string> lhs_terms, rhs_terms;

    split_equation_sides(equation, lhs, rhs);
    split_terms(lhs, lhs_terms);
    split_terms(rhs, rhs_terms);

    expand_all(lhs_terms);
    expand_all(rhs_terms);

    std::cout << "LHS\n";
    for (int i = 0; i < lhs_terms.size(); i++)
        std::cout << lhs_terms[i] << "\n";
    std::cout << "RHS\n";
    for (int i = 0; i < rhs_terms.size(); i++)
        std::cout << rhs_terms[i] << "\n";


    std::vector<pres> implicits;

    std::vector<std::vector<pres>> prefactor_vector;
    std::vector<std::vector<std::string>> fields_vector;

    if (lhs_terms[0].substr(0,2) == "dt")
    {
        dynamic = 1;
        field_name = lhs_terms[0].substr(2);
        if (!is_field(field_name))
        {
            std::cout << "Field not found: " << field_name << "\n";
            return -1;
        }
        if (is_field(field_name) > 1)
        {
            std::cout << "Nonlinearity in lhs: " << field_name << "\n";
            return -1;
        }
        lhs_terms.erase(lhs_terms.begin());
    }
    else
    {
        if (number_of_fields(lhs_terms[0]) != 1)
        {
            std::cout << "ERROR: Nonlinear term in left hand side: " << number_of_fields(lhs_terms[1]) << std::endl;
            std::cout << lhs_terms[0] << std::endl;
            return -1;
        }
        std::cout << "Getting field name\n";
        field_name = get_field_name(lhs_terms[0]);
        std::cout << field_name << "\n";
    }

    // process lhs
    for (int i = 0; i < lhs_terms.size(); i++)
    {
        std::vector<std::string> fields;
        get_field_vector(lhs_terms[i], fields);
        if (fields.size() != 1)
        {
            std::cout << "Nonlinear term in lhs\n";
            return -1;
        }
        if (fields[0] != field_name)
        {
            std::cout << "ERROR: " << field_name << " and " << fields[0] << " incompatible in lhs\n";
            return -1;
        }
        pres this_prefactor = get_prefactor(lhs_terms[i]);
        if (dynamic == 1) this_prefactor.preFactor *= -1.0f;
        implicits.push_back(this_prefactor);
    }

    // process rhs 
    for (int i = 0; i < rhs_terms.size(); i++)
    {
        std::vector<std::string> fields;
        get_field_vector(rhs_terms[i], fields);
        std::sort(fields.begin(), fields.end());

        pres this_prefactor = get_prefactor(rhs_terms[i]);
        // for (int j = 0; j < fields.size(); j++)
        // {
        //     std::cout << "Field " << j << " " << fields[j] << "\n";
        // }
        int exists = 0;
        for (int k = 0; k < fields_vector.size(); k++)
        {
            if (fields == fields_vector[k])
            {
                exists = 1;
                prefactor_vector[k].push_back(this_prefactor);
            }
        }
        if (exists == 0)
        {
            fields_vector.push_back(fields);
            prefactor_vector.push_back({this_prefactor});
        }
    }

    if (!(implicits.size() == 1 && implicits[0].preFactor == 1.0f && implicits[0].q2n == 0 && implicits[0].iqx == 0 && implicits[0].iqy == 0 && implicits[0].invq == 0))
    {
        for (int i = 0; i < implicits.size(); i++)
        {
            system->fieldsMap[field_name]->implicit.push_back(implicits[i]);
        }
    }

    if (prefactor_vector.size() != fields_vector.size())
    {
        std::cout << "Error processing equation\n";
        return -1;
    }
    for (int i = 0; i < fields_vector.size(); i++)
    {
        system->createTerm(field_name, prefactor_vector[i], fields_vector[i]);
    }

    return 0;
}
