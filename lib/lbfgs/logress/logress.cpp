#include <float.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

#include <lbfgs.h>
#include "quark.h"
#include "optparse.h"
#include "tokenize.h"

class option : public optparse
{
public:
    typedef std::vector<std::string> files_type;

    files_type  files;

    std::string mode;
    std::string model;
    std::string algorithm;
    std::string regularizer;
    int         maxiter;
    double      sigma;
    double      gamma;
    double      kappa;
    int         holdout;
    bool        cross_validation;

    option() :
        maxiter(1000), sigma(1), gamma(0.5), kappa(5),
        algorithm("log-likelihood"),
        holdout(-1), cross_validation(false)
    {
    }

    BEGIN_OPTION_MAP_INLINE()
        ON_OPTION(SHORTOPT('l') || LONGOPT("learn"))
            mode = "learn";

        ON_OPTION(SHORTOPT('t') || LONGOPT("tag"))
            mode = "tag";

        ON_OPTION_WITH_ARG(SHORTOPT('a') || LONGOPT("algorithm"))
            algorithm = arg;

        ON_OPTION_WITH_ARG(SHORTOPT('m') || LONGOPT("model"))
            model = arg;

        ON_OPTION_WITH_ARG(SHORTOPT('r') || LONGOPT("regularization"))
            regularizer = arg;

        ON_OPTION_WITH_ARG(SHORTOPT('i') || LONGOPT("maxiter"))
            maxiter = atoi(arg);

        ON_OPTION_WITH_ARG(SHORTOPT('s') || LONGOPT("sigma"))
            sigma = atof(arg);

        ON_OPTION_WITH_ARG(SHORTOPT('g') || LONGOPT("gamma"))
            gamma = atof(arg);

        ON_OPTION_WITH_ARG(SHORTOPT('k') || LONGOPT("kappa"))
            kappa = atof(arg);

        ON_OPTION_WITH_ARG(SHORTOPT('e') || LONGOPT("holdout"))
            holdout = atoi(arg);

        ON_OPTION(SHORTOPT('x') || LONGOPT("cross-validation"))
            cross_validation = true;

    END_OPTION_MAP()
};


typedef std::vector<int> content;

struct instance
{
    int label;
    int group;
    content cont;

    instance(int _label = 0, int _group = 0) : label(_label), group(_group)
    {
    }
};

typedef std::vector<instance> instances;

struct training
{
    instances& data;
    std::ostream& ls;

    double gamma;
    double kappa;
    double sigma2inv;
    int holdout;

    training(
        instances& _data,
        std::ostream& _ls
        )
        : data(_data), ls(_ls), holdout(-1), sigma2inv(0.)
    {
    }
};



static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    int i;
    lbfgsfloatval_t ll = 0.;
    training& tr = *reinterpret_cast<training*>(instance);

    // Initialize the gradient of every weight as zero.
    for (i = 0;i < n;++i) {
        g[i] = 0.;
    }

    // Loop over the instances.
    instances::const_iterator it;
    for (it = tr.data.begin();it != tr.data.end();++it) {
        double z = 0.;
        double d = 0.;

        // Exclude instances for holdout evaluation.
        if (it->group == tr.holdout) {
            continue;
        }

        // Compute the instance score.
        content::const_iterator itc;
        for (itc = it->cont.begin();itc != it->cont.end();++itc) {
            z += x[*itc];
        }

        if (z < -30.) {
            if (it->label) {
                d = 1.;
                ll += z;
            } else {
                d = 0.;
            }
        } else if (30. < z) {
            if (it->label) {
                d = 0.;
            } else {
                d = -1.;
                ll += (-z);
            }
        } else {
            double p = 1.0 / (1.0 + std::exp(-z));
            if (it->label) {
                d = 1.0 - p;
                ll += std::log(p);
            } else {
                d = -p;
                ll += std::log(1-p);                
            }
        }

        // Update the gradients for the weights.
        for (itc = it->cont.begin();itc != it->cont.end();++itc) {
            // Take the negatives of the gradients.
            g[*itc] -= d;
        }
    }

	/*
		L2 regularization.
		Note that we *add* the (weight * sigma) to g[i].
	 */
	if (tr.sigma2inv != 0.) {
        double norm = 0.;
		for (i = 0;i < n;++i) {
            g[i] += tr.sigma2inv * x[i];
            norm += x[i] * x[i];
		}
		ll -= (tr.sigma2inv * norm * 0.5);
	}

    // Minimize the negative of the log-likelihood.
    return -ll;
}

static lbfgsfloatval_t evaluate_max(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    int i;
    lbfgsfloatval_t ll = 0.;
    training& tr = *reinterpret_cast<training*>(instance);

    // Initialize the gradient of every weight as zero.
    for (i = 0;i < n;++i) {
        g[i] = 0.;
    }

    // Loop over the instances.
    instances::const_iterator it;
    for (it = tr.data.begin();it != tr.data.end();++it) {
        double z = -DBL_MAX;
        double d = 0.;

        // Exclude instances for holdout evaluation.
        if (it->group == tr.holdout) {
            continue;
        }

        // Compute the instance score.
        content::const_iterator itc;
        for (itc = it->cont.begin();itc != it->cont.end();++itc) {
            if (z < x[*itc]) {
                z = x[*itc];
            }
        }

        if (z < -30.) {
            if (it->label) {
                d = 1.;
                ll += z;
            } else {
                d = 0.;
            }
        } else if (30. < z) {
            if (it->label) {
                d = 0.;
            } else {
                d = -1.;
                ll += (-z);
            }
        } else {
            double p = 1.0 / (1.0 + std::exp(-z));
            if (it->label) {
                d = 1.0 - p;
                ll += std::log(p);
            } else {
                d = -p;
                ll += std::log(1-p);                
            }
        }

        // Update the gradients for the weights.
        for (itc = it->cont.begin();itc != it->cont.end();++itc) {
            if (z == x[*itc]) {
                // Take the negatives of the gradients.
                g[*itc] -= d;            
            }
        }
    }

	/*
		L2 regularization.
		Note that we *add* the (weight * sigma) to g[i].
	 */
	if (tr.sigma2inv != 0.) {
        double norm = 0.;
		for (i = 0;i < n;++i) {
            g[i] += tr.sigma2inv * x[i];
            norm += x[i] * x[i];
		}
		ll -= (tr.sigma2inv * norm * 0.5);
	}

    // Minimize the negative of the log-likelihood.
    return -ll;
}

static lbfgsfloatval_t evaluate_roc(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    int i;
    training& tr = *reinterpret_cast<training*>(instance);
    double r = tr.gamma;
    double k = tr.kappa;
    double u = 0.0;

    // Initialize the gradient of every weight as zero.
    for (i = 0;i < n;++i) {
        g[i] = 0.;
    }

    // Loop over the instances.
    instances::const_iterator it;
    for (it = tr.data.begin();it != tr.data.end();++it) {
        double d = 0., z = 0., s = 0., p = 0.;

        // Exclude instances for holdout evaluation.
        if (it->group == tr.holdout) {
            continue;
        }

        // Compute the instance score.
        content::const_iterator itc;
        for (itc = it->cont.begin();itc != it->cont.end();++itc) {
            z += x[*itc];
        }

        s = k * z;

        if (s < -30.) {
            p = 0.;
        } else if (30. < s) {
            p = 1.;
        } else {
            p = 1.0 / (1.0 + std::exp(-s));
        }

        if (it->label) {
            u += r * p;
            d = r * k * p * (1-p);
        } else {
            u += (1-r) * (1-p);
            d = -(1-r) * k * p * (1-p);
        }

        // Update the gradients for the weights.
        for (itc = it->cont.begin();itc != it->cont.end();++itc) {
            // Take the negatives of the gradients.
            g[*itc] -= d;
        }
    }

    return -u;
}

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
	int i, num_active_features = 0;
    training& tr = *reinterpret_cast<training*>(instance);
    std::ostream& os = tr.ls;

    for (i = 0;i < n;++i) {
        if (x[i] != 0.) ++num_active_features;
    }

    os << "***** Iteration #" << k << " *****" << std::endl;
    os << "Log-likelihood: " << -fx << std::endl;
    os << "Feature norm: " << xnorm << std::endl;
    os << "Error norm: " << gnorm << std::endl;
    os << "Active features: " << num_active_features << std::endl;
    os << "Line search trials: " << ls << std::endl;
    os << "Line search step: " << step << std::endl;

    // Holdout evaluation.
    if (tr.holdout != -1) {
        int eval[2][2] = {{0, 0}, {0, 0}};

        instances::const_iterator it;
        for (it = tr.data.begin();it != tr.data.end();++it) {
            if (it->group == tr.holdout) {
                // Score the instance for holdout evaluation.
                double z = 0.;
                content::const_iterator itc;
                for (itc = it->cont.begin();itc != it->cont.end();++itc) {
                    z += x[*itc];
                }

                // Tag the instance and update the confusion matrix.
                if (z <= 0.) {
                    ++eval[it->label][0];
                } else {
                    ++eval[it->label][1];
                }
            }
        }

        double accuracy = 0.;
        double precision = 0.;
        double recall = 0.;
        double f1score = 0.;
        int num_correct = eval[0][0] + eval[1][1];
        int num_total = eval[0][0] + eval[0][1] + eval[1][0] + eval[1][1];
        int num_system = (eval[0][1] + eval[1][1]);
        int num_positives = (eval[1][0] + eval[1][1]);

        if (0. < num_total) {
            accuracy = num_correct / (double)num_total;
        }
        if (0. < num_system) {
            precision = eval[1][1] / (double)num_system;
        }
        if (0. < num_positives) {
            recall = eval[1][1] / (double)num_positives;
        }
        if (0. < precision + recall) {
            f1score = 2 * precision * recall / (precision + recall);
        }

        os << "Accuracy: " << accuracy
            << " (" << num_correct << "/" << num_total << ")" << std::endl;
        os << "Precision: " << precision
            << " (" << eval[1][1] << "/" << num_system << ")" << std::endl;
        os << "Recall: " << recall
            << " (" << eval[1][1] << "/" << num_positives << ")" << std::endl;
        os << "F1 score: " << f1score << std::endl;
    }

    os << std::endl;

    return 0;
}

void read_data(std::istream& is, quark& features, instances& data, int group = 0)
{
    for (;;) {
        std::string line;
        std::getline(is, line);
        if (is.eof()) {
            break;
        }

        // Skip comment and empty lines.
        if (line.compare(0, 1, "#") == 0 || line.empty()) {
            continue;
        }

        // Initialize a tokenizer for a line.
        tokenizer token(line);

        // Read the label of the instance.
        if (!token.next()) {
            continue;            
        }

        // Create an instance.
        instance inst;
        inst.label = atoi(token->c_str()) == 1 ? 1 : 0;
        inst.group = group;

        // Read the features in the line.
        while (token.next()) {
            int fid = features[*token];
            inst.cont.push_back(fid);
        }

        // Append the instance to the data set.
        data.push_back(inst);
    }
}

int learn(instances& data, quark& features, int holdout, option& opt)
{
    int i, status, ret = 0;
    std::ostream& os = std::cout;

    // Create a callback instance for the L-BFGS solver.
    training tr(data, os);
    tr.holdout = holdout;

    // L-BFGS parameter object.
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);

    // Regularization parameters.
    if (opt.regularizer == "l1") {
        param.orthantwise_c = 1.0 / opt.sigma;
        tr.sigma2inv = 0;
    } else if (opt.regularizer == "l2") {
        param.orthantwise_c = 0.;
        tr.sigma2inv = 1.0 / (opt.sigma * opt.sigma);
    }

    tr.gamma = opt.gamma;
    tr.kappa = opt.kappa;

    // L-BFGS optimization parameters.
    param.max_iterations = opt.maxiter;
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;

    // Allocate an array for feature weights.
    double *w = new double[features.size()];
    for (i = 0;i < features.size();++i) {
        w[i] = 0;
    }

    // 
    os << "Number of instances: " << data.size() << std::endl;
    os << "Number of features: " << features.size() << std::endl;
    os << "regularization: " << opt.regularizer << std::endl;
    os << "regularization.sigma: " << opt.sigma << std::endl;
    os << "lbfgs.num_memories: " << param.m << std::endl;
    os << "lbfgs.max_iterations: " << param.max_iterations << std::endl;
    os << "lbfgs.epsilon: " << param.epsilon << std::endl;
    os << std::endl;

    // Call the L-BFGS solver.
    if (opt.algorithm == "log-likelihood") {
        status = lbfgs(
            features.size(),
            w,
            NULL,
            evaluate,
            progress,
            &tr,
            &param
            );
    } else if (opt.algorithm == "max") {
        status = lbfgs(
            features.size(),
            w,
            NULL,
            evaluate_max,
            progress,
            &tr,
            &param
            );
    } else if (opt.algorithm == "roc") {
        status = lbfgs(
            features.size(),
            w,
            NULL,
            evaluate_roc,
            progress,
            &tr,
            &param
            );
    }
    if (status == 0) {
        os << "L-BFGS resulted in convergence" << std::endl;
    } else {
        os << "L-BFGS terminated with code (" << status << ")" << std::endl;
    }
    os << std::endl;

    // Store the feature weights.
    //  "<weight>\t<feature>\n" (one feature per line).
    if (!opt.model.empty()) {
        std::ofstream ofs(opt.model.c_str());
        if (ofs.fail()) {
            os << "ERROR: failed to store the model: " << opt.model << std::endl;
            ret = 1;
            goto error_exit;

        } else {
            os << "Writing the model: " << opt.model << std::endl;
            os << std::endl;

            for (i = 0;i < features.size();++i) {
                // Output features with non-zero weights.
                if (w[i] != 0.) {
                    ofs << w[i] << '\t' << features.to_string(i) << std::endl;
                }
            }
        }
    }

error_exit:
    delete[] w;
    return ret;
}

int learn_main(option& opt)
{
    int i, ret;
    int num_groups = 0;
    quark features;
    instances data;
    std::ostream& os = std::cout;

    // Read training data from file(s) or STDIN.
    if (!opt.files.empty()) {
        for (i = 0;i < opt.files.size();++i) {
            const std::string& fn = opt.files[i];
            std::ifstream ifs(fn.c_str());
            if (!ifs.fail()) {
                os  << "Reading the data set "
                    << "(" << i+1 << "/" << opt.files.size() << "): "
                    << fn << std::endl;
                read_data(ifs, features, data, i);
                ifs.close();
                ++num_groups;
            } else {
                os << "ERROR: failed to open a data set: " << fn << std::endl;
            }
        }
    } else {
        os << "Reading a data set from STDIN" << std::endl;
        read_data(std::cin, features, data);
    }
    os << std::endl;

    if (opt.cross_validation) {
        for (i = 0;i < num_groups;++i) {
            os << "===== Cross validation "
                << i << "/" << num_groups << " =====" << std::endl;
            learn(data, features, i, opt);
            os << std::endl;
        }
    } else {
        learn(data, features, opt.holdout, opt);
    }

    return 0;
}



#define	APPLICATION_S	"Logistic Regression (LogRess)"
#define	VERSION_S		"0.1"
#define	COPYRIGHT_S		"Copyright (c) 2008 Naoaki Okazaki"

int main(int argc, char *argv[])
{
    option opt;
    int i, ret, arg_used;
    std::ostream& os = std::cout;

    os << APPLICATION_S " " VERSION_S "  " COPYRIGHT_S << std::endl;
    os << std::endl;

    // Parse the command-line options.
    try { 
        arg_used = opt.parse(argv, argc);
    } catch (const optparse::unrecognized_option& e) {
        os << "Unrecognized option: " << e.what() << std::endl;
        return 1;
    } catch (const optparse::invalid_value& e) {
        os << "Invalid value: " << e.what() << std::endl;
        return 1;
    }

    // Store file names for input data.
    for (i = arg_used;i < argc;++i) {
        opt.files.push_back(argv[i]);
    }

    // Process the data.
    if (opt.mode == "learn") {
        ret = learn_main(opt);
    } else if (opt.mode == "tag") {

    } else {

    }

    return ret;
}
