/*
 * Heat conduction on a square plate for t = [0,1]
 *
 * Author: Changyu Meng
 */

#include <deal.II/grid/tria.h>              // Triangulation class
#include <deal.II/dofs/dof_handler.h>       // Association of degrees of freedom to vertices, lines, and cells
#include <deal.II/grid/grid_generator.h>    // Generate standard grids
#include <deal.II/grid/grid_in.h>           // Read grid from disk
#include <deal.II/grid/grid_out.h>          // Output grids in various graphics formats
#include <deal.II/grid/manifold_lib.h>      // Describe the boundary of a circular domain
#include <deal.II/grid/grid_refinement.h>   // Decide which cells to flag for refinement
#include <deal.II/dofs/dof_tools.h>         // Manipulating dof
#include <deal.II/dofs/dof_renumbering.h>   // Renumber dof
#include <deal.II/fe/fe_values.h>           // Assembling the matrix using quadrature on each cell
#include <deal.II/fe/fe_q.h>                // Lagrange elements, one dof on each vertex of the triangulation
#include <deal.II/base/quadrature_lib.h>    // Quadratures
#include <deal.II/base/function.h>          // Treatment of boundary values
#include <deal.II/base/smartpointer.h>      // Make sure objects are not deleted while they are still in use
#include <deal.II/base/logstream.h>         // Suppress the unwanted output from the linear solvers
#include <deal.II/base/convergence_table.h> // Collect all important data during a run
#include <deal.II/base/utilities.h>
#include <deal.II/numerics/error_estimator.h> // Compute the refinement indicator
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>            // Visualize the pattern of nonzero entries resulting from the distribution of dof on the grid
#include <deal.II/lac/dynamic_sparsity_pattern.h> // Intermediate sparsity pattern structure
#include <deal.II/lac/solver_cg.h>                // Solver
#include <deal.II/lac/precondition.h>             // Preconditioner useful for CG solver
#include <deal.II/lac/affine_constraints.h>       // Conform constraints for hanging nodes
#include <fstream>
#include <iostream>

#define EPS 1e-12

namespace Heat
{
    using namespace dealii;

    template <int dim>
    class HeatProblem
    {
    public:
        enum RefinementMode
        {
            global_refinement,
            adaptive_refinement
        };

        HeatProblem(const FiniteElement<dim> &fe,
                    const RefinementMode refinement_mode,
                    const double diffusity);
        void run();

    private:
        void make_grid(); // Preprocessing, could also deal with B.C., geometries, etc.
        void read_grid();
        void refine_grid();
        void setup_system(); // Setup the data structures
        void assemble_system();
        void solve_time_step();
        void output_results(const unsigned int cycle) const;

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

        const RefinementMode refinement_mode;
        ConvergenceTable convergence_table;

        double diffusity;
        // double time;
        // double time_step;
        // unsigned int timestep_number;

        // const double theta;
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
        // double return_value = 0.0;
        // for (unsigned int i = 0; i < dim; ++i)
        //     return_value += 4.0 * std::pow(p(i), 4.0);

        double return_value = 0.0;

        return return_value;
    }

    template <int dim>
    double BoundaryValues<dim>::value(const Point<dim> &p,
                                      const unsigned int /*component*/) const
    {
        if (fabs(p(1) - 1.0) < EPS)
            return 1.0;
        else
            return 0.0;
    }

    template <int dim>
    HeatProblem<dim>::HeatProblem(const FiniteElement<dim> &fe,
                                  const RefinementMode refinement_mode,
                                  const double diffusity)
        : dof_handler(triangulation), fe(&fe), refinement_mode(refinement_mode), diffusity(diffusity) {}

    template <int dim>
    void HeatProblem<dim>::make_grid()
    {
        // triangulation is a collection of cells

        GridGenerator::hyper_cube(triangulation, -1, 1);
        triangulation.refine_global(5);

        std::cout << "   Number of active cells: " << triangulation.n_active_cells()
                  << std::endl
                  << "   Total number of cells: " << triangulation.n_cells()
                  << std::endl;

        /*
        const Point<2> center(1, 0);
        const double inner_radius = 0.5, outer_radius = 1.0;
        GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius, 10);

        for (unsigned int step = 0; step < 3; ++step)
        {
            // Active cells are those that are not further refined, and the only ones
            // that can be marked for further refinement.
            for (auto &cell : triangulation.active_cell_iterators())
            {
                // Loop over all vertices of the cells.
                for (const auto v : cell->vertex_indices())
                {
                    const double distance_from_center = center.distance(cell->vertex(v));
                    if (std::fabs(distance_from_center - inner_radius) <= 1e-6 * inner_radius)
                    {
                        cell->set_refine_flag(); // mark the cell that needs refinement
                        break;
                    }
                }
            }
            triangulation.execute_coarsening_and_refinement();
        }
        */

        std::ofstream out("square-grid.svg");
        GridOut grid_out;
        grid_out.write_svg(triangulation, out);
        std::cout << "Grid written to square-grid.svg" << std::endl;
    }

    template <int dim>
    void HeatProblem<dim>::read_grid()
    {
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation);
        std::ifstream input_file("circle-grid.inp");
        Assert(dim == 2, ExcInternalError());
        grid_in.read_ucd(input_file);
    }

    template <int dim>
    void HeatProblem<dim>::setup_system()
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
    void HeatProblem<dim>::assemble_system()
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
                for (const unsigned int i : fe_values.dof_indices())
                {
                    for (const unsigned int j : fe_values.dof_indices())
                    {
                        cell_matrix(i, j) +=
                            (diffusity *
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
    void HeatProblem<dim>::solve_time_step()
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
    void HeatProblem<dim>::refine_grid()
    {
        switch (refinement_mode)
        {
        case global_refinement:
        {
            triangulation.refine_global(1);
            break;
        }
        case adaptive_refinement:
        {
            Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
            KellyErrorEstimator<dim>::estimate(dof_handler,
                                               QGauss<dim - 1>(fe->degree + 1),
                                               std::map<types::boundary_id, const Function<dim> *>(),
                                               solution,
                                               estimated_error_per_cell);
            GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                            estimated_error_per_cell,
                                                            0.3,
                                                            0.03);
            triangulation.execute_coarsening_and_refinement();
            break;
        }
        default:
        {
            Assert(false, ExcNotImplemented());
        }
        }
    }

    template <int dim>
    void HeatProblem<dim>::output_results(const unsigned int cycle) const
    {
        { // Edges can appear to be curved in gnuplot
            GridOut grid_out;
            std::ofstream output("grid-" + std::to_string(cycle) + ".gnuplot");
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

            std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
            data_out.write_vtu(output);
        }
    }

    template <int dim>
    void HeatProblem<dim>::run()
    {
        std::cout << "Solving problem in " << dim << " space dimensions."
                  << std::endl;

        const unsigned int n_cycles =
            (refinement_mode == global_refinement) ? 5 : 9;

        // Use external grid
        // read_grid();
        // const SphericalManifold<dim> boundary;
        // triangulation.set_all_manifold_ids_on_boundary(0);
        // triangulation.set_manifold(0, boundary);

        for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
        {
            std::cout << "Cycle " << cycle << ':' << std::endl;

            if (cycle == 0)
            {
                GridGenerator::hyper_cube(triangulation, -1, 1);
                triangulation.refine_global(1);
            }
            else
                refine_grid();
            // if (cycle != 0)
            //     refine_grid(); // For global refinement, use triangulation.refine_global(1);

            std::cout << "   Number of active cells: "
                      << triangulation.n_active_cells()
                      << std::endl
                      << "   Total number of cells: "
                      << triangulation.n_cells()
                      << std::endl;

            setup_system();
            assemble_system();
            solve_time_step();
            output_results(cycle);
        }
    }
} // namespace Heat

int main()
{
    using namespace Heat;
    const unsigned int dim = 2;

    try
    {
        FE_Q<dim> fe(1);
        double diffusity = 0.1;
        HeatProblem<dim> heat_problem_2d(fe, HeatProblem<dim>::global_refinement, diffusity);
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