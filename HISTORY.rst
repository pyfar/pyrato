=======
History
=======

1.0.0 (2026-03-19)
------------------

.. warning::

    ``pyrato`` version 1.0.0 introduces several breaking changes which are not backwards compatible with prior versions.
    This includes changes in the module structure, and deprecations of functions that are now part of the ``pyfar`` v0.8 package.


Breaking changes
^^^^^^^^^^^^^^^^

- ``pyrato`` has been structured into the modules

  - ``pyrato.analytic`` containing analytic functions for room acoustics.
  - ``pyrato.dsp`` containing functions for low-level room impulse response processing.
  - ``pyrato.edc`` containing functions for computing Energy Decay Curves (EDC).
  - ``pyrato.parameters`` containing functions to compute room acoustical parameters from simple geometric considerations.
  - ``pyrato.parametric`` containing functions to compute room acoustical parameters from room impulse responses.

- The following functions were removed and calling functions now use equivalent functions from pyfar

  - ``pyrato.dsp.find_impulse_response_start`` was removed in favor of ``pyfar.dsp.find_impulse_response_start``.
  - ``pyrato.dsp.center_frequencies_octaves`` and ``pyrato.dsp.center_frequencies_third_octaves`` were removed in favor of ``pyfar.constants.fractional_octave_frequencies_nominal``.
  - ``pyrato.dsp.filter_fractional_octave_bands`` was removed in favor of ``pyfar.dsp.filter.fractional_octave_bands``.
  - ``pyrato.dsp.time_shift`` was removed in favor of ``pyfar.dsp.time_shift``.
  - ``pyrator.parametric.air_attenuation_coefficient`` was removed in favor of ``pyfar.constants.air_attenuation``.
  - ``pyrato.parametric.calculate_speed_of_sound`` was removed in favor of multiple functions in ``pyfar.constants`` that can compute the speed of sound.

- ``pyrato.dsp.truncate_energy_decay_curve`` was renamed to ``pyrato.dsp.threshold_energy_decay_curve```.
- ``pyrato.parametric.reverberation_time_eyring`` was refactored for parameters to be more verbose and consistent with other functions in pyrato.
- ``pyrato.parametric.calculate_sabine_reverberation_time`` was renamed to ``pyrato.parametric.reverberation_time_sabine`` and refactored for parameters to be more verbose and consistent with other functions in pyrato.

Added:
^^^^^^

- Room acoustic parameters according to ISO 3382 to ``pyrator.parameters``

  - ``clarity`` to compute the clarity for arbitrary time limits including the standardized parameters C50 and C80.
  - ``early_lateral_energy_fraction`` to compute the ear lateral energy fraction J_LF.
  - ``late_lateral_sound_level`` to compute the late lateral sound level L_J
  - ``sound_strength`` to compute the sound sound_strength G.
  - ``speech_transmission_index_indirect`` and ``modulation_transfer_function`` to compute the STI based on impulse responses.

- ``pyrato.edc.intersection_time_lundby`` can now process impulse response channels independently and by default raises a warning, if the computation fails for a specific channel. Before this update, it raised an error, even if the computation only failed for certain channels.

Changed:
^^^^^^^^

- ``pyrato.parametric.energy_decay_curve_analytic`` was refactored to take the desired reverberation time as input parameter.
- Updated tests and package versions used for testing, as well as test fixtures.

Documentation
^^^^^^^^^^^^^
- Improved the scope and clarity of the documentation.
- Added buttons to copy code contained in the documentation.


0.4.2 (2026-03-18)
------------------
* Bugfix: Channel independent normalization parametrization was reversed (PR #117)
* Bugfix: Reverberation time estimation of non normalized EDCs (PR #138)
* Bugfix: Properly support EDC calculation for multi-dimensional RIRs (PR #113)
* Bugfix: Properly support multi-dimensional RIRs in the Schroeder integration (PR #62)
* Dependencies: Limit to pyfar < 0.8.0 (PR #161)
* CI: Add deprecation tests


0.4.1 (2024-06-05)
------------------
* fix: include analytic module in the package (PR #27)
* fix: allow T15 reverberation time estimation in ``reverberation_time_linear_regression`` (PR #51)
* fix: several deprecation warnings (PR #54, #55, #56)
* fix: Correct index selection in truncation time estimation (#32)
* fix: Fix deprecation warnings in example notebook (#58)
* doc: apply Documentation style changes to match with pyfar (PR #30, #34, #48, #61)
* ci: apply pyfar's packaging style (PR #49)
* ci: change from flake8 to ruff and apply pyfar rule set (PR #52)
* ci: configure and change to bumpversion for release (PR #60)



0.4.0 (2024-03-20)
------------------
* Use pyfar audio objects to store RIRs and EDCs (PR #7)
* Drop support for Python 3.7 (PR #14)
* rename RT estimation function to `reverberation_time_linear_regression` (PR #13)
* use pyfar's impulse response start finding function and fractional octave filters, respective functions are deprecated and will be removed in pyrato 0.5.0 #17  (PR #7)
* Add ISO 3382 compliant dynamic range threshold to `energy_decay_curve_truncation`, add similar functionality to `energy_decay_curve_chu` (PR #18)
* Bugfix for wrong truncation time for multichannel RIRs (PR #19)
* Update of the documentation style and adaption of the shared gallery homepage (PR #22)

0.3.2 (2022-10-10)
------------------
* Hotfix for deprecated generation of nested ragged sequences in Lundeby's intersection time algorithm.

0.3.1 (2022-05-31)
------------------
* Bugfix Lundeby's intersection time algorithm

0.3.0 (2021-06-12)
------------------
* Release on PyPI as pyrato

0.2.1 (2020-02-26)
------------------

* Bugfix Chu's EDC calculation
* Documentation fixes
* Improved identification of impulse response onset

0.2.0 (2019-11-19)
------------------

* Energy decay curve calculation with various noise compensation methods
* Linear regression based reverberation time estimation

0.1.0 (2018-12-13)
------------------

* First release
