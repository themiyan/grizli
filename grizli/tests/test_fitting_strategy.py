import unittest
import tempfile
import os
import numpy as np
import shutil

from grizli.pipeline import auto_script
from grizli import fitting
from grizli import utils # For PLINE, BOUNDED_DEFAULTS if needed by generate_fit_params defaults

# Mock IGM for tests if needed, to avoid dependency on eazy.igm if not installed/configured
class IGMMock:
    def full_IGM(self, z, lambda_obs):
        return np.ones_like(lambda_obs)

fitting.IGM = IGMMock()


class TestFittingStrategy(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create dummy Prep and Extractions directories if auto_script functions expect them
        os.makedirs("Prep", exist_ok=True)
        os.makedirs("Extractions", exist_ok=True)

    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    def test_stack_pa_1d_workflow(self):
        """
        Test the 'stack_p_a_1d' workflow, focusing on parameter generation
        and basic invocation of the specialized function.
        """
        field_root = 'test_stack_pa'
        object_id = 1
        
        # Part 1: Test generate_fit_params
        fit_args_file = os.path.join(self.temp_dir, f"{field_root}_fit_args.npy")
        
        # Minimal params for generate_fit_params
        # generate_fit_params has many default arguments, let's rely on them
        # but ensure crucial ones for the test are set.
        # Need to check which ones are strictly required if not using default_params from YAML
        
        # Use a simplified pline for testing to avoid WCS issues if not needed
        simple_pline = {"kernel": "point", "pixfrac": 0.2, "pixscale": 0.1, "size": 8}

        generated_args = auto_script.generate_fit_params(
            field_root=field_root,
            MW_EBV=0.01,
            zr=[0.1, 0.3], # Smaller range for faster test if it were to run
            dz=[0.005, 0.001],
            fwhm=1000,
            fit_strategy='stack_p_a_1d',
            save_file=fit_args_file,
            pline=simple_pline, 
            # The following are defaults in generate_fit_params, but let's be explicit
            # for parameters that _run_all_stack_p_a_1d might use from the loaded args.
            fitter=['nnls', 'bounded'],
            fit_only_beams=True, # This is a default in generate_fit_params
            run_fit=True, # This is a default in generate_fit_params
            poly_order=7, # Default
            fsps=True, # Default
            min_sens=0.01, # Default
            sys_err=0.03, # Default
            fcontam=0.2, # Default
            include_photometry=False, # To avoid needing phot_obj
            use_phot_obj=False, # To avoid needing phot_obj
            fit_trace_shift=False # Default
        )

        self.assertTrue(os.path.exists(fit_args_file))
        
        loaded_args = np.load(fit_args_file, allow_pickle=True)[0]
        self.assertEqual(loaded_args['fit_strategy'], 'stack_p_a_1d')
        self.assertEqual(loaded_args['field_root'], field_root)

        # Part 2: Test direct call to _run_all_stack_p_a_1d (conceptual for now)
        # Since actual beam files are not present, it should exit early.
        
        # Prepare kwargs for _run_all_stack_p_a_1d
        # These would normally be a mix of defaults from run_all and loaded from fit_args_file
        dummy_kwargs = loaded_args.copy() # Start with generated args
        dummy_kwargs['id'] = object_id 
        dummy_kwargs['group_name'] = field_root # Typically same as field_root for this strategy
        dummy_kwargs['root'] = field_root # For finding beams files
        dummy_kwargs['file_pattern'] = '{root}_{id:05d}' # Default
        dummy_kwargs['save_figures'] = False # Don't try to save figures in test
        dummy_kwargs['write_fits_files'] = True # Test file writing logic
        dummy_kwargs['verbose'] = False # Keep test output clean
        
        # Ensure necessary templates are in dummy_kwargs or use defaults
        if 't0' not in dummy_kwargs:
            dummy_kwargs['t0'] = utils.load_templates(line_complexes=True, fsps_templates=True, fwhm=dummy_kwargs['fwhm'])
        if 't1' not in dummy_kwargs:
            dummy_kwargs['t1'] = utils.load_templates(line_complexes=False, fsps_templates=True, fwhm=dummy_kwargs['fwhm'])

        # Call the internal function
        # Expecting (None, None, None, None, None) or (mb, None, None, pa_spectra_1d, None) if no beams files
        # As per current implementation, if no beams_files, it returns (None, None, None, None, None)
        # If it proceeds, mb would be an empty MultiBeam if no files, or a MultiBeam object
        # The pa_spectra_1d would be empty if no PAs or no valid spectra.
        # The fit_result_1d and tfit_1d would be None if no valid combined spectrum.
        # The line_hdul_1d would be None if no valid fit.
        
        # Since field_root is 'test_stack_pa', it won't find 'test_stack_pa_00001.beams.fits'
        # So, the early exit for "No beams files found" should be triggered.
        expected_return_on_no_beams = (None, None, None, None, None)
        
        # Change root to something that definitely won't exist to ensure it fails on glob
        dummy_kwargs_no_beams = dummy_kwargs.copy()
        dummy_kwargs_no_beams['root'] = "dummy_nonexistent_root"
        
        fit_outputs = fitting._run_all_stack_p_a_1d(object_id, **dummy_kwargs_no_beams)
        
        self.assertEqual(fit_outputs, expected_return_on_no_beams, 
                         "Function should return specific tuple on not finding beams files.")

        # Conceptual: If we had mock data or could mock MultiBeam loading
        # We would then assert on the creation of .full.fits and .1D.fits
        # For now, this part is skipped as _run_all_stack_p_a_1d will exit early.
        
        # Example of what assertions would look like if files were created:
        # full_fits_name = f"{dummy_kwargs['group_name']}_{object_id:05d}.full.fits"
        # oned_fits_name = f"{dummy_kwargs['group_name']}_{object_id:05d}.1D.fits"
        # self.assertTrue(os.path.exists(full_fits_name))
        # self.assertTrue(os.path.exists(oned_fits_name))
        
        # with pyfits.open(full_fits_name) as hdul:
        #     self.assertIn('ZFIT_1D', hdul)
        #     self.assertIn('TEMPL_1D', hdul)
        #     self.assertIn('COVAR_1D', hdul)

if __name__ == '__main__':
    unittest.main()
