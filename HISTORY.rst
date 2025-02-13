=======
History
=======

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
