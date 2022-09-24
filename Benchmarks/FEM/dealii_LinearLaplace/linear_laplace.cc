/*
 * Solve a linear Laplace equation
 *
 * Author: Changyu Meng
 */

#include <deal.II/base/smartpointer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#define EPS 1e-12

namespace LinearLaplace
{
    using namespace dealii;

    template <int dim>
    class LinearLaplaceProblem
    {
    public:
        LinearLaplaceProblem(const FiniteElement<dim> &fe);
        void run();

    private:
        void make_grid(const unsigned int cycle); // Preprocessing, could also deal with B.C., geometries, etc.
        void setup_system();                      // Setup the data structures
        void assemble_system();
        void solve_time_step();
        void output_results() const;

        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        SmartPointer<const FiniteElement<dim>> fe;

        AffineConstraints<double> constraints; // Holds a list of constraints for the hanging nodes and the boundary conditions

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> mass_matrix;
        SparseMatrix<double> laplace_matrix;
        SparseMatrix<double> system_matrix;

        Vector<double> solution;
        Vector<double> old_solution;
        Vector<double> system_rhs;
    };

    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p,
                             const unsigned int component = 0) const override;
    };

    template <int dim>
    class BoundaryValues : public Function<dim>
    {
    public:
        virtual double value(const Point<dim> &p,
                             const unsigned int component = 0) const override;
    };

    template <int dim>
    double RightHandSide<dim>::value(const Point<dim> &p,
                                     const unsigned int /*component*/) const
    {
        double return_value = 1.0;
        for (unsigned int i = 0; i < dim; ++i)
            return_value += 0.0 * std::pow(p(i), 4.0);

        return return_value; // Return 1.0
    }

    template <int dim>
    double BoundaryValues<dim>::value(const Point<dim> &p,
                                      const unsigned int /*component*/) const
    {
        if (fabs(p(1) - 1.0) < EPS)
            return 0.0;
        else
            return 0.0; // Return 0.0
    }

    template <int dim>
    double coefficient(const Point<dim> &p)
    {
        if (p.square() < 0.5 * 0.5)
            return 20;
        else
            return 1;
    }

    template <int dim>
    LinearLaplaceProblem<dim>::LinearLaplaceProblem(const FiniteElement<dim> &fe)
        : dof_handler(triangulation), fe(&fe) {}

    template <int dim>
    void LinearLaplaceProblem<dim>::make_grid(const unsigned int cycle)
    {
        // Triangulation is a collection of cells

        GridGenerator::hyper_cube(triangulation, -1, 1);
        triangulation.refine_global(cycle);

        std::cout << "   Number of active cells: " << triangulation.n_active_cells()
                  << std::endl
                  << "   Total number of cells: " << triangulation.n_cells()
                  << std::endl;
    }

    template <int dim>
    void LinearLaplaceProblem<dim>::setup_system()
    {
        dof_handler.distribute_dofs(*fe);
        DoFRenumbering::Cuthill_McKee(dof_handler); // Renumber the dof immediately after distributing, since the
                                                    // dof number will affect hanging nodes, sparsity patterns etc.
        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                  << std::endl;

        constraints.clear(); // First clear the current set of constraints from the last system
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        constraints.close();

        DynamicSparsityPattern dsp(dof_handler.n_dofs()); // A special data structure that can copy
                                                          // into SparsityPattern object without much overhead.
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        constraints.condense(dsp);
        sparsity_pattern.copy_from(dsp); // SparsityPattern object only stores the places where entries are

        std::ofstream out("sparsity_pattern.svg");
        sparsity_pattern.print_svg(out);

        system_matrix.reinit(sparsity_pattern); // Entries of matrix is stored in SparseMatrix object
        solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());
    }

    template <int dim>
    void LinearLaplaceProblem<dim>::assemble_system()
    {
        // Loop over all cells
        // Mapping from reference cell to real cell
        QGauss<dim> quadrature_formula(fe->degree + 1);
        FEValues<dim> fe_values(*fe,
                                quadrature_formula,
                                update_values | update_gradients | update_quadrature_points | update_JxW_values);
        const RightHandSide<dim> right_hand_side;
        const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);
            cell_matrix = 0.;
            cell_rhs = 0.;

            for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
                const double current_coef = coefficient(fe_values.quadrature_point(q_index));
                for (const unsigned int i : fe_values.dof_indices())
                {
                    for (const unsigned int j : fe_values.dof_indices())
                    {
                        cell_matrix(i, j) +=
                            (current_coef *
                             fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                             fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                             fe_values.JxW(q_index));           // dx
                    }

                    const auto &x_q = fe_values.quadrature_point(q_index);
                    cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                                    right_hand_side.value(x_q) *        // f(x_q)
                                    fe_values.JxW(q_index));            // dx
                }
            }
            cell->get_dof_indices(local_dof_indices);

            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                    system_matrix.add(local_dof_indices[i],
                                      local_dof_indices[j],
                                      cell_matrix(i, j));

                system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }
        }
        constraints.condense(system_matrix);
        constraints.condense(system_rhs);

        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 BoundaryValues<dim>(),
                                                 boundary_values); // Work on faces that have been marked
                                                                   // with boundary indicator 0
        MatrixTools::apply_boundary_values(boundary_values,
                                           system_matrix,
                                           solution,
                                           system_rhs);
    }

    template <int dim>
    void LinearLaplaceProblem<dim>::solve_time_step()
    {
        // SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());
        SolverControl solver_control(1000, 1e-12);
        SolverCG<Vector<double>> solver(solver_control);

        PreconditionSSOR<SparseMatrix<double>> preconditioner;
        preconditioner.initialize(system_matrix, 1.2);

        solver.solve(system_matrix, solution, system_rhs, preconditioner);
        constraints.distribute(solution); // Assign correct values to the hanging nodes

        std::cout << "   " << solver_control.last_step()
                  << " CG iterations needed to obtain convergence." << std::endl;
    }

    template <int dim>
    void LinearLaplaceProblem<dim>::output_results() const
    {
        { // Edges can appear to be curved in gnuplot
            GridOut grid_out;
            std::ofstream output("grid.gnuplot");
            GridOutFlags::Gnuplot gnuplot_flags(false, 5);
            grid_out.set_flags(gnuplot_flags);
            MappingQ<dim> mapping(3);
            grid_out.write_gnuplot(triangulation, output, &mapping);
        }
        {
            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(solution, "solution");
            data_out.build_patches();

            std::ofstream output("solution.vtu");
            data_out.write_vtu(output);
        }
    }

    template <int dim>
    void LinearLaplaceProblem<dim>::run()
    {
        std::cout << "Solving problem in " << dim << " space dimensions."
                  << std::endl;

        const unsigned int n_cycles = 6;
        make_grid(n_cycles);
        setup_system();
        assemble_system();
        solve_time_step();
        output_results();
    }
} // namespace LinearLaplace

int main()
{
    using namespace LinearLaplace;
    const unsigned int dim = 2;

    try
    {
        FE_Q<dim> fe(1);
        LinearLaplaceProblem<dim> heat_problem_2d(fe);
        heat_problem_2d.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}