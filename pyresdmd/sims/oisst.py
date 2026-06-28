# this is mostly written by Claude Opus 4.8, Claude Fable 5 and GPT 5.5 through Codex

from pathlib import Path
import sys

def _find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "pyresdmd").exists():
            return path
    raise RuntimeError("Could not find repo root containing pyproject.toml and pyresdmd/")

REPO_ROOT = _find_repo_root(Path.cwd())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("Found pyresdmd root.")

from pathlib import Path
import csv
import random

import numpy as np
import zlib
import torch
import time

from pyresdmd.compute.spectra import (
	compute_eigendecomposition_from_weights,
	compute_koopman_modes,
	compute_residuals,
	EDMD,
	compute_pseudospectra,
)
from pyresdmd.dicts.hermite_dictionary import HermiteDictionary
from pyresdmd.dicts.nn.relu.ReLUDictionary import ReLUDictionary
from pyresdmd.dicts.polynomial_dictionary import PolynomialDictionary
from pyresdmd.dicts.tensor_product_dictionary import TensorProductDictionary
from pyresdmd.dicts.trainable_dictionary import TrainableDictionary
from pyresdmd.utils.helpers import benchmark_metrics
from pyresdmd.utils.plotters import plot_curve, plot_pseudospec

from matplotlib import rc

rc('font', **{'family': 'serif', 'size' : 16, 'serif': ['Computer Modern']})
rc('text', usetex=True)

class HybridFixedReLUDictionary(torch.nn.Module):
	'''
		Concatenates fixed constant/state observables with trainable ReLU observables.
	'''
	def __init__(self, input_dim : int, n_functions : int, hidden_dim : int = 32, hidden_layers : int = 2,
		include_constant : bool = True, include_identity : bool = True) -> None:
		super().__init__()
		self._input_dim = input_dim
		self.include_constant = include_constant
		self.include_identity = include_identity
		self.trainable = ReLUDictionary(
			input_dim = input_dim,
			n_functions = n_functions,
			hidden_dim = hidden_dim,
			hidden_layers = hidden_layers,
		)

	@property
	def input_dim(self) -> int:
		return self._input_dim

	@property
	def size(self) -> int:
		size = self.trainable.size
		if self.include_constant:
			size += 1
		if self.include_identity:
			size += self.input_dim
		return size

	def evaluate(self, x : torch.Tensor) -> torch.Tensor:
		features = []
		if self.include_constant:
			features.append(torch.ones((x.shape[0], 1), device = x.device, dtype = x.dtype))
		if self.include_identity:
			features.append(x)
		features.append(self.trainable.evaluate(x))
		return torch.cat(features, dim = 1)

def _set_random_seed(seed : int | None) -> None:
	'''
		Sets the random seeds used by this benchmark.
	'''
	if seed is None:
		return
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

def _oisst_data_dir(data_dir : Path | None = None) -> Path:
	'''
		Resolves the OISST data directory.
	'''
	return Path(__file__).resolve().parents[1] / 'data' if data_dir is None else Path(data_dir)

def _key_in_sample(key : str, sample_fraction : float, seed : int) -> bool:
	'''
		Deterministically decides whether a row key belongs to the sample.

		Uses a hash of (seed, key) so both CSV files agree on the sampled rows
		without either file needing to be fully loaded.
	'''
	digest = zlib.crc32(f"{seed}|{key}".encode())
	return digest / 0xFFFFFFFF < sample_fraction


def _load_csv_table(filename : str, data_dir : Path | None = None) -> tuple[list[str], np.ndarray]:
	'''
		Loads a CSV file as a header list and a string table.
	'''
	data_dir = _oisst_data_dir(data_dir)

	path = data_dir / filename
	with open(path, newline = '') as file:
		headers = next(csv.reader(file))

	table = np.genfromtxt(path, delimiter = ',', skip_header = 1, dtype = str)
	if table.ndim == 1:
		table = table[None, :]

	return headers, table


def _mode_columns(headers : list[str]) -> list[tuple[int, str]]:
	'''
		Returns the columns that represent comparable modes, preserving file order.
	'''
	metadata_columns = {'time', 'lat', 'lon', 'index'}
	return [(index, header) for index, header in enumerate(headers) if header.lower() not in metadata_columns]

def _load_oisst_eofs(eofs_filename : str = 'eofs.csv', data_dir : Path | None = None) -> tuple[np.ndarray, np.ndarray]:
	'''
		Loads OISST EOF grid coordinates and EOF loadings.
	'''
	data_dir = _oisst_data_dir(data_dir)
	eofs_table = np.genfromtxt(data_dir / eofs_filename, delimiter = ',', skip_header = 1)
	if eofs_table.ndim == 1:
		eofs_table = eofs_table[None, :]
	if eofs_table.shape[1] < 4:
		raise ValueError(f"{eofs_filename} must contain an index, lat, lon, and at least one EOF mode")

	return eofs_table[:, 1:3], eofs_table[:, 3:]

def _field_on_oisst_grid(lat_lon : np.ndarray, field : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	'''
		Places a long-form OISST field on a rectangular lat/lon grid.
	'''
	lats = np.unique(lat_lon[:, 0])
	lons = np.unique(lat_lon[:, 1])
	lats.sort()
	lons.sort()

	grid = np.full((lats.shape[0], lons.shape[0]), np.nan, dtype = field.dtype)
	lat_lookup = {lat: index for index, lat in enumerate(lats)}
	lon_lookup = {lon: index for index, lon in enumerate(lons)}

	for (lat, lon), value in zip(lat_lon, field):
		grid[lat_lookup[lat], lon_lookup[lon]] = value

	return lats, lons, grid

def _unique_plot_filename(filename : str) -> str:
	'''
		Returns a filename that does not overwrite an existing plot.
	'''
	path = Path(filename)
	if not path.exists():
		return filename

	stem = path.stem
	suffix = path.suffix
	parent = path.parent
	counter = 1
	while True:
		candidate = parent / f"{stem}_{counter}{suffix}"
		if not candidate.exists():
			return str(candidate)
		counter += 1

def _oisst_mode_fields(modes : torch.Tensor | np.ndarray,
	eofs_filename : str = 'eofs.csv',
	data_dir : Path | None = None,
	grid_shape : tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	'''
		Projects Koopman mode coefficients onto the same spatial grid used for OISST plots.
	'''
	if isinstance(modes, torch.Tensor):
		modes_array = modes.detach().cpu().numpy()
	else:
		modes_array = np.asarray(modes)

	if modes_array.ndim == 1:
		modes_array = modes_array[None, :]
	if modes_array.ndim != 2:
		raise ValueError("modes must have shape (n_modes, d) or (d,)")

	if grid_shape is not None:
		if np.prod(grid_shape) != modes_array.shape[1]:
			raise ValueError("grid_shape does not match the Koopman mode length")
		return np.arange(grid_shape[0]), np.arange(grid_shape[1]), modes_array.reshape((-1, *grid_shape))

	try:
		eof_lat_lon, eof_modes = _load_oisst_eofs(eofs_filename, data_dir = data_dir)
		if modes_array.shape[1] == eof_lat_lon.shape[0]:
			fields = modes_array
		else:
			shared_modes = min(modes_array.shape[1], eof_modes.shape[1])
			if shared_modes == 0:
				raise ValueError('No shared EOF modes are available for plotting')
			fields = modes_array[:, :shared_modes] @ eof_modes[:, :shared_modes].T
		lats, lons, first_grid = _field_on_oisst_grid(eof_lat_lon, fields[0])
		field_grids = np.empty((fields.shape[0], first_grid.shape[0], first_grid.shape[1]), dtype = fields.dtype)
		field_grids[0] = first_grid
		for index in range(1, fields.shape[0]):
			_, _, field_grids[index] = _field_on_oisst_grid(eof_lat_lon, fields[index])
		return lats, lons, field_grids
	except OSError:
		raise

def _mode_agreement_matrix(reference_fields : np.ndarray, comparison_fields : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	'''
		Computes pairwise phase/sign-invariant agreement and subspace angle matrices.
	'''
	agreements = np.zeros((reference_fields.shape[0], comparison_fields.shape[0]), dtype = float)
	angles = np.full_like(agreements, np.nan)

	for i, reference_field in enumerate(reference_fields):
		reference_vector = reference_field.reshape(-1)
		for j, comparison_field in enumerate(comparison_fields):
			comparison_vector = comparison_field.reshape(-1)
			mask = np.isfinite(reference_vector) & np.isfinite(comparison_vector)
			if not np.any(mask):
				continue
			u = reference_vector[mask]
			v = comparison_vector[mask]
			denominator = np.linalg.norm(u) * np.linalg.norm(v)
			if denominator == 0:
				continue
			agreement = np.abs(np.vdot(u, v)) / denominator
			agreement = float(np.clip(agreement, 0.0, 1.0))
			agreements[i, j] = agreement
			angles[i, j] = np.degrees(np.arccos(agreement))

	return agreements, angles

def _nino34_reference_field(lats : np.ndarray, lons : np.ndarray,
	lat_bounds : tuple[float, float] = (-5.0, 5.0),
	lon_bounds : tuple[float, float] = (190.0, 240.0),
) -> np.ndarray:
	'''
		Builds a simple El Nino reference pattern over the Nino 3.4 region.

		Longitudes are compared on a 0-360 grid, so lon_bounds=(190, 240)
		corresponds to 170W-120W.
	'''
	lon_360 = np.mod(lons, 360.0)
	lat_mask = (lats >= min(lat_bounds)) & (lats <= max(lat_bounds))
	lon_start, lon_end = np.mod(lon_bounds, 360.0)
	if lon_start <= lon_end:
		lon_mask = (lon_360 >= lon_start) & (lon_360 <= lon_end)
	else:
		lon_mask = (lon_360 >= lon_start) | (lon_360 <= lon_end)

	if not np.any(lat_mask) or not np.any(lon_mask):
		raise ValueError("Nino 3.4 reference region does not overlap the OISST grid")

	field = np.zeros((lats.shape[0], lons.shape[0]), dtype = float)
	field[np.ix_(lat_mask, lon_mask)] = 1.0
	return field

def _phase_align_field(field : np.ndarray, reference_field : np.ndarray) -> np.ndarray:
	'''
		Rotates a complex field so its projection onto reference_field is positive.
	'''
	field_vector = field.reshape(-1)
	reference_vector = reference_field.reshape(-1)
	mask = np.isfinite(field_vector) & np.isfinite(reference_vector)
	if not np.any(mask):
		return field

	inner = np.vdot(field_vector[mask], reference_vector[mask])
	if np.abs(inner) == 0:
		return field
	return (inner / np.abs(inner)) * field

def _residual_at(residuals : torch.Tensor | np.ndarray | None, index : int) -> float | None:
	'''
		Returns a Python residual value for title formatting.
	'''
	if residuals is None:
		return None
	residuals_array = residuals.detach().cpu().numpy() if isinstance(residuals, torch.Tensor) else np.asarray(residuals)
	if residuals_array.ndim != 1 or not 0 <= index < residuals_array.shape[0]:
		return None
	return float(np.real(residuals_array[index]))

def locate_oisst_el_nino_mode(modes : torch.Tensor | np.ndarray,
	eofs_filename : str = 'eofs.csv',
	data_dir : Path | None = None,
	grid_shape : tuple[int, int] | None = None,
	nino_lat_bounds : tuple[float, float] = (-5.0, 5.0),
	nino_lon_bounds : tuple[float, float] = (190.0, 240.0),
	max_modes : int | None = None,
	residuals : torch.Tensor | np.ndarray | None = None,
	use_residual_penalty : bool = True,
	residual_penalty_power : float = 1.0,
) -> dict:
	'''
		Finds the Koopman mode whose spatial field best matches the Nino 3.4 box.

		The score is abs(<mode, reference>) / (||mode|| ||reference||), making the
		selection invariant to Koopman-mode sign or complex phase. If residuals are
		provided and use_residual_penalty is True, the spatial agreement is divided
		by (1 + residual)^p so highly residual modes are less likely to be reported
		as the El Nino node.
	'''
	lats, lons, fields = _oisst_mode_fields(modes, eofs_filename, data_dir, grid_shape)
	if max_modes is not None:
		fields = fields[:max_modes]
	if fields.shape[0] == 0:
		raise ValueError("At least one mode is required to locate the El Nino mode")

	reference_field = _nino34_reference_field(lats, lons, nino_lat_bounds, nino_lon_bounds)
	agreements, angles = _mode_agreement_matrix(reference_field[None, :, :], fields)
	scores = agreements[0].copy()
	residual_values = None
	if residuals is not None:
		residual_values = residuals.detach().cpu().numpy() if isinstance(residuals, torch.Tensor) else np.asarray(residuals)
		residual_values = np.real(residual_values).astype(float)
		residual_values = residual_values[:fields.shape[0]]
		if residual_values.shape[0] == fields.shape[0] and use_residual_penalty:
			valid_residuals = np.isfinite(residual_values) & (residual_values >= 0.0)
			penalties = np.ones_like(scores)
			penalties[valid_residuals] = (1.0 + residual_values[valid_residuals]) ** residual_penalty_power
			scores = np.divide(scores, penalties, out = np.zeros_like(scores), where = penalties > 0)
		elif residual_values.shape[0] != fields.shape[0]:
			residual_values = None
	mode_index = int(np.argmax(scores))
	return {
		'mode_index': mode_index,
		'agreement': float(agreements[0, mode_index]),
		'angle': float(angles[0, mode_index]),
		'score': float(scores[mode_index]),
		'residual': None if residual_values is None else float(residual_values[mode_index]),
		'lats': lats,
		'lons': lons,
		'fields': fields,
		'reference_field': reference_field,
		'agreements': agreements[0],
		'angles': angles[0],
		'scores': scores,
		'used_residual_penalty': bool(use_residual_penalty and residual_values is not None),
	}

def _greedy_mode_matches(agreements : np.ndarray) -> list[tuple[int, int]]:
	'''
		Builds a simple one-to-one matching by taking highest agreements first.
	'''
	matches = []
	used_rows = set()
	used_cols = set()
	for flat_index in np.argsort(agreements, axis = None)[::-1]:
		row, col = np.unravel_index(flat_index, agreements.shape)
		if row in used_rows or col in used_cols:
			continue
		matches.append((int(row), int(col)))
		used_rows.add(row)
		used_cols.add(col)
		if len(used_rows) == agreements.shape[0] or len(used_cols) == agreements.shape[1]:
			break
	return matches

def plot_oisst_mode_agreement(reference_modes : torch.Tensor | np.ndarray,
	comparison_modes : torch.Tensor | np.ndarray,
	reference_name : str = 'Trained ReLU',
	comparison_name : str = 'Polynomial',
	reference_mode_index : int = 0,
	eofs_filename : str = 'eofs.csv',
	data_dir : Path | None = None,
	grid_shape : tuple[int, int] | None = None,
	cmap : str = 'RdBu_r',
	save : bool = True,
	show : bool = False,
	filename : str | None = None,
	title : str | None = None,
	max_modes : int | None = None,
):
	'''
		Compares two sets of OISST Koopman modes using vectorized plotted fields.

		Agreement is abs(<u, v>) / (||u|| ||v||), so parallel modes have score 1
		and subspace angle 0 degrees. The absolute value makes the score invariant
		to sign and complex phase.
	'''
	import matplotlib.pyplot as plt

	lats, lons, reference_fields = _oisst_mode_fields(reference_modes, eofs_filename, data_dir, grid_shape)
	_, _, comparison_fields = _oisst_mode_fields(comparison_modes, eofs_filename, data_dir, grid_shape)

	if max_modes is not None:
		reference_fields = reference_fields[:max_modes]
		comparison_fields = comparison_fields[:max_modes]

	if not 0 <= reference_mode_index < reference_fields.shape[0]:
		raise ValueError(f"reference_mode_index must be between 0 and {reference_fields.shape[0] - 1}")

	agreements, angles = _mode_agreement_matrix(reference_fields, comparison_fields)
	matches = _greedy_mode_matches(agreements)
	best_comparison_index = int(np.argmax(agreements[reference_mode_index]))

	reference_field = reference_fields[reference_mode_index]
	comparison_field = comparison_fields[best_comparison_index]
	reference_vector = reference_field.reshape(-1)
	comparison_vector = comparison_field.reshape(-1)
	mask = np.isfinite(reference_vector) & np.isfinite(comparison_vector)
	phase_factor = 1.0 + 0.0j
	if np.any(mask):
		inner = np.vdot(comparison_vector[mask], reference_vector[mask])
		if np.abs(inner) > 0:
			phase_factor = inner / np.abs(inner)
	aligned_comparison_field = phase_factor * comparison_field
	difference_field = reference_field - aligned_comparison_field

	fig = plt.figure(figsize = (13, 8))
	grid = fig.add_gridspec(2, 3, height_ratios = [1.0, 1.15])
	top_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
	matrix_ax = fig.add_subplot(grid[1, :])

	fields_to_plot = [
		(f"{reference_name} mode {reference_mode_index}", np.real(reference_field)),
		(f"{comparison_name} mode {best_comparison_index}", np.real(aligned_comparison_field)),
		("Real difference", np.real(difference_field)),
	]
	for ax, (axis_title, values) in zip(top_axes, fields_to_plot):
		limit = np.nanmax(np.abs(values))
		vmin, vmax = (-limit, limit) if np.isfinite(limit) and limit > 0 else (None, None)
		mesh = ax.pcolormesh(lons, lats, values, shading = 'auto', cmap = cmap, vmin = vmin, vmax = vmax)
		ax.set_title(axis_title)
		ax.set_xlabel('Longitude')
		ax.set_ylabel('Latitude')
		fig.colorbar(mesh, ax = ax)

	matrix = matrix_ax.imshow(agreements, origin = 'lower', aspect = 'auto', vmin = 0.0, vmax = 1.0, cmap = 'viridis')
	matrix_ax.set_xlabel(f"{comparison_name} mode")
	matrix_ax.set_ylabel(f"{reference_name} mode")
	matrix_ax.set_title("Mode agreement: cos(subspace angle)")
	for row, col in matches:
		matrix_ax.plot(col, row, marker = 's', markersize = 10, markerfacecolor = 'none', markeredgecolor = 'white', markeredgewidth = 1.5)
	matrix_ax.plot(best_comparison_index, reference_mode_index, marker = 'o', markersize = 11, markerfacecolor = 'none', markeredgecolor = 'red', markeredgewidth = 1.5)
	fig.colorbar(matrix, ax = matrix_ax, label = 'agreement')

	best_agreement = agreements[reference_mode_index, best_comparison_index]
	best_angle = angles[reference_mode_index, best_comparison_index]
	if title is None:
		title = f"{reference_name} vs {comparison_name}: mode {reference_mode_index} best match {best_comparison_index}, agreement={best_agreement:.3f}, angle={best_angle:.1f} deg"
	fig.suptitle(title)
	fig.tight_layout()

	if save:
		if filename is None:
			filename = f'oisst_koopman_mode_agreement_{int(time.time())}.png'
		filename = _unique_plot_filename(filename)
		fig.savefig(filename, dpi = 200, bbox_inches = 'tight')

	if show:
		plt.show()
	else:
		plt.close(fig)

	return {
		'fig': fig,
		'axes': [*top_axes, matrix_ax],
		'agreements': agreements,
		'angles': angles,
		'matches': matches,
		'best_comparison_index': best_comparison_index,
		'best_agreement': best_agreement,
		'best_angle': best_angle,
		'filename': filename if save else None,
	}

def plot_oisst_el_nino_mode_comparison(trained_modes : torch.Tensor | np.ndarray,
	init_modes : torch.Tensor | np.ndarray,
	poly_modes : torch.Tensor | np.ndarray,
	trained_residuals : torch.Tensor | np.ndarray | None = None,
	init_residuals : torch.Tensor | np.ndarray | None = None,
	poly_residuals : torch.Tensor | np.ndarray | None = None,
	eofs_filename : str = 'eofs.csv',
	data_dir : Path | None = None,
	grid_shape : tuple[int, int] | None = None,
	cmap : str = 'RdBu_r',
	save : bool = True,
	show : bool = False,
	filename : str | None = None,
	title : str | None = None,
	max_modes : int | None = None,
	use_residual_penalty : bool = True,
	residual_penalty_power : float = 1.0,
) -> dict:
	'''
		Plots the trained El Nino mode and the closest untrained ReLU/poly modes.

		The trained El Nino mode is selected by agreement with the Nino 3.4 box.
		The other two modes are selected by agreement with that trained spatial mode.
	'''
	import matplotlib.pyplot as plt

	located = locate_oisst_el_nino_mode(
		trained_modes,
		eofs_filename = eofs_filename,
		data_dir = data_dir,
		grid_shape = grid_shape,
		max_modes = max_modes,
		residuals = trained_residuals,
		use_residual_penalty = use_residual_penalty,
		residual_penalty_power = residual_penalty_power,
	)
	lats = located['lats']
	lons = located['lons']
	trained_fields = located['fields']
	trained_index = located['mode_index']
	reference_field = located['reference_field']

	_, _, init_fields = _oisst_mode_fields(init_modes, eofs_filename, data_dir, grid_shape)
	_, _, poly_fields = _oisst_mode_fields(poly_modes, eofs_filename, data_dir, grid_shape)
	if max_modes is not None:
		init_fields = init_fields[:max_modes]
		poly_fields = poly_fields[:max_modes]

	init_agreements, init_angles = _mode_agreement_matrix(trained_fields[trained_index][None, :, :], init_fields)
	poly_agreements, poly_angles = _mode_agreement_matrix(trained_fields[trained_index][None, :, :], poly_fields)
	init_index = int(np.argmax(init_agreements[0]))
	poly_index = int(np.argmax(poly_agreements[0]))

	trained_field = _phase_align_field(trained_fields[trained_index], reference_field)
	init_field = _phase_align_field(init_fields[init_index], trained_field)
	poly_field = _phase_align_field(poly_fields[poly_index], trained_field)

	plot_values = [
		np.real(trained_field),
		np.real(init_field),
		np.real(poly_field),
	]
	normalized_plot_values = []
	for values in plot_values:
		limit = np.nanmax(np.abs(values))
		if np.isfinite(limit) and limit > 0:
			normalized_plot_values.append(values / limit)
		else:
			normalized_plot_values.append(values)

	trained_residual = _residual_at(trained_residuals, trained_index)
	init_residual = _residual_at(init_residuals, init_index)
	poly_residual = _residual_at(poly_residuals, poly_index)

	def _residual_text(value : float | None) -> str:
		return 'n/a' if value is None else f'{value:.3g}'

	fig, axes = plt.subplots(1, 3, figsize = (15, 4.5), squeeze = False)
	axes = axes[0]
	axis_titles = [
		f"Trained ReLU\nresidual={_residual_text(trained_residual)}",
		f"Untrained ReLU\nresidual={_residual_text(init_residual)}",
		f"Polynomial\nresidual={_residual_text(poly_residual)}",
	]

	for ax, values, axis_title in zip(axes, normalized_plot_values, axis_titles):
		mesh = ax.pcolormesh(lons, lats, values, shading = 'auto', cmap = cmap, vmin = -1.0, vmax = 1.0)
		ax.set_title(axis_title)
		ax.set_xlabel('Longitude')
		ax.set_ylabel('Latitude')
		fig.colorbar(mesh, ax = ax, label = 'Normalized amplitude')

	if title is not None:
		fig.suptitle(title)
	fig.tight_layout()

	if save:
		if filename is None:
			filename = f'oisst_el_nino_mode_comparison_{int(time.time())}.png'
		filename = _unique_plot_filename(filename)
		fig.savefig(filename, dpi = 200, bbox_inches = 'tight')

	if show:
		plt.show()
	else:
		plt.close(fig)

	return {
		'fig': fig,
		'axes': axes,
		'filename': filename if save else None,
		'trained_mode_index': trained_index,
		'init_mode_index': init_index,
		'poly_mode_index': poly_index,
		'trained_nino34_agreement': located['agreement'],
		'trained_nino34_angle': located['angle'],
		'init_agreement': float(init_agreements[0, init_index]),
		'init_angle': float(init_angles[0, init_index]),
		'poly_agreement': float(poly_agreements[0, poly_index]),
		'poly_angle': float(poly_angles[0, poly_index]),
		'trained_residual': trained_residual,
		'init_residual': init_residual,
		'poly_residual': poly_residual,
	}

def plot_oisst_koopman_mode(modes : torch.Tensor | np.ndarray, mode_index : int,
	eigenvalues : torch.Tensor | np.ndarray | None = None,
	eofs_filename : str = 'eofs.csv',
	data_dir : Path | None = None,
	component : str = 'both',
	grid_shape : tuple[int, int] | None = None,
	cmap : str = 'RdBu_r',
	save : bool = True,
	show : bool = False,
	filename : str | None = None,
	title : str | None = None,
	figsize : tuple[float, float] | None = None,
):
	'''
		Plots a Koopman mode for the OISST example.

		In the OISST workflow, states are usually EOF coefficients. In that case this
		function projects the Koopman mode coefficients back to the SST grid using
		eofs.csv before plotting. If the selected mode already has one value per EOF
		grid point, it is plotted directly. For non-OISST vectors, pass grid_shape.

		Parameters
		------------------------------
		modes : torch.Tensor | np.ndarray
			Koopman modes with shape (n_modes, d), as returned by
			compute_koopman_modes(... )['modes'].
		mode_index : int
			Zero-based index of the Koopman mode to plot.
		eigenvalues : torch.Tensor | np.ndarray | None
			Optional eigenvalues used only for the plot title.
		eofs_filename : str
			EOF file containing columns index, lat, lon, EOF modes.
		data_dir : Path | None
			Directory containing eofs_filename. Defaults to pyresdmd/data.
		component : str
			One of 'real', 'imag', 'abs', 'phase', or 'both'. 'both' plots real and
			imaginary parts side by side.
		grid_shape : tuple[int, int] | None
			Optional direct reshape target for modes that are already spatial vectors.
		cmap : str
			Matplotlib colormap.
		save : bool
			If True, saves the figure.
		show : bool
			If True, displays the figure.
		filename : str | None
			Output filename. If None and save is True, a timestamped PNG is used.
		title : str | None
			Optional figure title.
		figsize : tuple[float, float] | None
			Optional Matplotlib figure size.

		Returns
		------------------------------
		fig, axes
			The Matplotlib figure and axes.
	'''
	import matplotlib.pyplot as plt

	if component not in {'real', 'imag', 'abs', 'phase', 'both'}:
		raise ValueError("component must be one of 'real', 'imag', 'abs', 'phase', or 'both'")

	if isinstance(modes, torch.Tensor):
		modes_array = modes.detach().cpu().numpy()
	else:
		modes_array = np.asarray(modes)

	if modes_array.ndim == 1:
		modes_array = modes_array[None, :]
	if modes_array.ndim != 2:
		raise ValueError("modes must have shape (n_modes, d) or (d,)")
	if not 0 <= mode_index < modes_array.shape[0]:
		raise ValueError(f"mode_index must be between 0 and {modes_array.shape[0] - 1}")

	mode = modes_array[mode_index]
	lat_lon = None

	try:
		eof_lat_lon, eof_modes = _load_oisst_eofs(eofs_filename, data_dir = data_dir)
		if mode.shape[0] == eof_lat_lon.shape[0]:
			field = mode
			lat_lon = eof_lat_lon
		else:
			shared_modes = min(mode.shape[0], eof_modes.shape[1])
			if shared_modes == 0:
				raise ValueError('No shared EOF modes are available for plotting')
			field = eof_modes[:, :shared_modes] @ mode[:shared_modes]
			lat_lon = eof_lat_lon
	except OSError:
		if grid_shape is None:
			raise
		if np.prod(grid_shape) != mode.shape[0]:
			raise ValueError("grid_shape does not match the selected Koopman mode length")
		field = mode.reshape(grid_shape)

	if lat_lon is not None:
		lats, lons, field_grid = _field_on_oisst_grid(lat_lon, field)
	else:
		lats = np.arange(field.shape[0])
		lons = np.arange(field.shape[1])
		field_grid = field

	component_values = {
		'real': ('Real part', np.real(field_grid)),
		'imag': ('Imaginary part', np.imag(field_grid)),
		'abs': ('Magnitude', np.abs(field_grid)),
		'phase': ('Phase', np.angle(field_grid)),
	}
	components_to_plot = ['real', 'imag'] if component == 'both' else [component]

	if figsize is None:
		figsize = (11, 4) if len(components_to_plot) == 2 else (6, 4)
	fig, axes = plt.subplots(1, len(components_to_plot), figsize = figsize, squeeze = False)
	axes = axes[0]

	eigenvalue_text = ''
	if eigenvalues is not None:
		eigenvalues_array = eigenvalues.detach().cpu().numpy() if isinstance(eigenvalues, torch.Tensor) else np.asarray(eigenvalues)
		if eigenvalues_array.ndim == 1 and mode_index < eigenvalues_array.shape[0]:
			eigenvalue_text = f", lambda={eigenvalues_array[mode_index]:.3g}"

	if title is None:
		title = f"OISST Koopman mode {mode_index}{eigenvalue_text}"

	for ax, comp in zip(axes, components_to_plot):
		comp_title, values = component_values[comp]
		if comp in {'real', 'imag'}:
			limit = np.nanmax(np.abs(values))
			vmin, vmax = (-limit, limit) if np.isfinite(limit) and limit > 0 else (None, None)
		else:
			vmin, vmax = None, None

		mesh = ax.pcolormesh(lons, lats, values, shading = 'auto', cmap = cmap, vmin = vmin, vmax = vmax)
		ax.set_title(comp_title)
		ax.set_xlabel('Longitude')
		ax.set_ylabel('Latitude')
		fig.colorbar(mesh, ax = ax)

	fig.suptitle(title)
	fig.tight_layout()

	if save:
		if filename is None:
			filename = f'oisst_koopman_mode_{mode_index}_{int(time.time())}.png'
		fig.savefig(filename, dpi = 200, bbox_inches = 'tight')

	if show:
		plt.show()
	else:
		plt.close(fig)

	return fig, axes

def load_sampled_baseline(baseline_filename : str = 'reconstructed_baseline.csv', eofs_filename : str = 'eofs.csv', dates : np.ndarray | None = None, sample_fraction : float = 0.2, seed : int = 0, data_dir : Path | None = None) -> dict:
	'''
		Streams the baseline reconstruction once and keeps a random sample of rows,
		indexed against the EOF grid and the prediction dates. The result lets the
		reconstruction L2 error be computed in memory for every repeat without ever
		writing reconstructed_signal.csv.
	'''
	if not 0.0 < sample_fraction <= 1.0:
		raise ValueError('sample_fraction must be in the interval (0, 1]')

	if data_dir is None:
		data_dir = Path(__file__).resolve().parents[1] / 'data'

	if dates is None:
		dates = oisst_dates()[:-1]

	eofs_table = np.genfromtxt(data_dir / eofs_filename, delimiter = ',', skip_header = 1)
	if eofs_table.ndim == 1:
		eofs_table = eofs_table[None, :]
	if eofs_table.shape[1] < 4:
		raise ValueError(f"{eofs_filename} must contain an index, lat, lon, and at least one EOF mode")

	lat_lon = eofs_table[:, 1:3]
	eof_modes = eofs_table[:, 3:]

	grid_lookup = {(str(float(lat)), str(float(lon))): index for index, (lat, lon) in enumerate(lat_lon)}
	date_lookup = {date: index for index, date in enumerate(dates)}

	rng = random.Random(seed)
	t_indices = []
	g_indices = []
	baseline_values = []

	with open(data_dir / baseline_filename, newline = '') as file:
		reader = csv.reader(file)
		headers = next(reader, [])

		try:
			time_index = headers.index('time')
			lat_index = headers.index('lat')
			lon_index = headers.index('lon')
		except ValueError as error:
			raise ValueError(f"{baseline_filename} must contain time, lat, and lon columns") from error

		value_index = next((index for index, header in enumerate(headers) if header.lower() not in {'time', 'lat', 'lon', 'index'}), None)
		if value_index is None:
			raise ValueError(f"No value column found in {baseline_filename}")

		for row in reader:
			if rng.random() > sample_fraction:
				continue

			t_idx = date_lookup.get(row[time_index])
			g_idx = grid_lookup.get((str(float(row[lat_index])), str(float(row[lon_index]))))
			if t_idx is None or g_idx is None:
				continue

			value = float(row[value_index])
			if not np.isfinite(value):
				continue

			t_indices.append(t_idx)
			g_indices.append(g_idx)
			baseline_values.append(value)

	if not t_indices:
		raise ValueError('Sampling selected no matching rows; increase sample_fraction or check that the baseline grid matches eofs.csv')

	return {
		't_indices': np.asarray(t_indices, dtype = np.int64),
		'g_indices': np.asarray(g_indices, dtype = np.int64),
		'baseline_values': np.asarray(baseline_values, dtype = np.float64),
		'eof_modes': eof_modes,
	}

def reconstruction_relative_l2_error_from_modes(predicted_modes : np.ndarray, sampled_baseline : dict) -> float:
	'''
		Computes the sampled relative L2 reconstruction error directly from predicted
		mode coefficients, evaluating the reconstruction only at the sampled points.
	'''
	eof_modes = sampled_baseline['eof_modes']
	shared_modes = min(predicted_modes.shape[1], eof_modes.shape[1])
	if shared_modes == 0:
		raise ValueError('No shared EOF modes between the predictions and eofs.csv')

	predicted_values = np.einsum(
		'ij,ij->i',
		predicted_modes[sampled_baseline['t_indices'], :shared_modes],
		eof_modes[sampled_baseline['g_indices'], :shared_modes],
	)
	baseline_values = sampled_baseline['baseline_values']

	valid = np.isfinite(predicted_values)
	predicted_values = predicted_values[valid]
	baseline_values = baseline_values[valid]

	denominator = float(np.dot(baseline_values, baseline_values))
	if np.isclose(denominator, 0.0):
		raise ValueError('Reference reconstruction has zero norm')

	difference = predicted_values - baseline_values
	return float(np.sqrt(np.dot(difference, difference) / denominator) * 100.0)


def mean_percentage_error(reference_filename : str, prediction_filename : str, mode : int | str, data_dir : Path | None = None) -> float:
	'''
		Computes the mean absolute percentage error for a given mode column between two CSV files.

		The mode can be provided as a 1-based mode index among the comparable data columns or as a
		column name shared by both CSV files.
	'''
	if data_dir is None:
		data_dir = Path(__file__).resolve().parents[1] / 'data'

	reference_headers, reference_table = _load_csv_table(reference_filename, data_dir = data_dir)
	prediction_headers, prediction_table = _load_csv_table(prediction_filename, data_dir = data_dir)

	if reference_table.shape[0] != prediction_table.shape[0]:
		raise ValueError(
			f"{reference_filename} has {reference_table.shape[0]} rows but {prediction_filename} has {prediction_table.shape[0]} rows"
		)

	reference_modes = _mode_columns(reference_headers)
	prediction_modes = _mode_columns(prediction_headers)

	if isinstance(mode, str):
		if mode not in reference_headers:
			raise ValueError(f"Column '{mode}' not found in {reference_filename}")
		if mode not in prediction_headers:
			raise ValueError(f"Column '{mode}' not found in {prediction_filename}")
		reference_index = reference_headers.index(mode)
		prediction_index = prediction_headers.index(mode)
	else:
		if mode <= 0:
			raise ValueError('mode must be positive')
		if mode > len(reference_modes):
			raise ValueError(f"{reference_filename} only has {len(reference_modes)} comparable mode columns")
		if mode > len(prediction_modes):
			raise ValueError(f"{prediction_filename} only has {len(prediction_modes)} comparable mode columns")
		reference_index = reference_modes[mode - 1][0]
		prediction_index = prediction_modes[mode - 1][0]

	reference_values = reference_table[:, reference_index].astype(float)
	prediction_values = prediction_table[:, prediction_index].astype(float)

	valid_rows = np.isfinite(reference_values) & np.isfinite(prediction_values) & ~np.isclose(reference_values, 0.0)
	if not np.any(valid_rows):
		raise ValueError('No valid rows available to compute percentage error')

	percentage_errors = np.abs(prediction_values[valid_rows] - reference_values[valid_rows]) / np.abs(reference_values[valid_rows]) * 100.0
	return float(np.mean(percentage_errors))

def reconstruction_relative_l2_error(reference_filename : str, prediction_filename : str, sample_fraction : float = 0.2, seed : int = 0, data_dir : Path | None = None) -> float:
	'''
		Computes a normalized reconstruction error between two long-form CSV files.

		Both files are streamed: only the sampled reference rows are held in memory
		(as floats), and the prediction file is never stored at all.
	'''
	if not 0.0 < sample_fraction <= 1.0:
		raise ValueError('sample_fraction must be in the interval (0, 1]')

	if data_dir is None:
		data_dir = Path(__file__).resolve().parents[1] / 'data'

	reference_path = data_dir / reference_filename
	prediction_path = data_dir / prediction_filename
	metadata_columns = {'time', 'lat', 'lon', 'index'}
	key_columns = ['time', 'lat', 'lon']

	# Pass 1: stream the reference file, keeping only sampled rows as floats.
	sampled_reference : dict[str, list[float]] = {}
	with open(reference_path, newline = '') as reference_file:
		reader = csv.reader(reference_file)
		reference_headers = next(reader, [])

		try:
			key_indices = [reference_headers.index(column) for column in key_columns]
		except ValueError as error:
			raise ValueError(f"{reference_filename} is missing one of the key columns {key_columns}") from error

		value_indices = [index for index, header in enumerate(reference_headers) if header.lower() not in metadata_columns]
		if not value_indices:
			raise ValueError('No comparable value columns found in the reconstructed CSV files')

		for row in reader:
			key = '|'.join(row[index] for index in key_indices)
			if not _key_in_sample(key, sample_fraction, seed):
				continue
			sampled_reference[key] = [float(row[index]) for index in value_indices]

	if not sampled_reference:
		raise ValueError('Sampling selected no rows; increase sample_fraction')

	# Pass 2: stream the prediction file and accumulate sums; store nothing.
	reference_sum = 0.0
	error_sum = 0.0
	matched_rows = 0

	with open(prediction_path, newline = '') as prediction_file:
		reader = csv.reader(prediction_file)
		prediction_headers = next(reader, [])

		if prediction_headers != reference_headers:
			raise ValueError(
				f"{reference_filename} and {prediction_filename} must have the same columns to compare reconstructed output"
			)

		for row in reader:
			key = '|'.join(row[index] for index in key_indices)
			reference_values = sampled_reference.pop(key, None)
			if reference_values is None:
				continue

			matched_rows += 1
			for reference_value, value_index in zip(reference_values, value_indices):
				prediction_value = float(row[value_index])
				if not (np.isfinite(reference_value) and np.isfinite(prediction_value)):
					continue
				reference_sum += reference_value * reference_value
				error_value = prediction_value - reference_value
				error_sum += error_value * error_value

	if matched_rows == 0:
		raise ValueError('No matching rows found between the reconstructed CSV files')

	if np.isclose(reference_sum, 0.0):
		raise ValueError('Reference reconstruction has zero norm')

	return float(np.sqrt(error_sum / reference_sum) * 100.0)

def old_reconstruction_relative_l2_error(reference_filename : str, prediction_filename : str, sample_fraction : float = 0.2, seed : int = 0, data_dir : Path | None = None) -> float:
	'''
		Computes a normalized reconstruction error between two long-form CSV files.

		The metric is computed on a random sample of matching rows to keep memory usage low:
		||prediction - reference||_2 / ||reference||_2, expressed as a percentage.
		This is more stable than percentage error when the reconstructed field crosses zero.
	'''
	if not 0.0 < sample_fraction <= 1.0:
		raise ValueError('sample_fraction must be in the interval (0, 1]')

	if data_dir is None:
		data_dir = Path(__file__).resolve().parents[1] / 'data'

	reference_path = data_dir / reference_filename
	prediction_path = data_dir / prediction_filename
	metadata_columns = {'time', 'lat', 'lon', 'index'}
	rng = random.Random(seed)
	key_columns = ['time', 'lat', 'lon']

	with open(reference_path, newline = '') as reference_file, open(prediction_path, newline = '') as prediction_file:
		reference_reader = csv.DictReader(reference_file)
		prediction_reader = csv.DictReader(prediction_file)

		reference_headers = reference_reader.fieldnames or []
		prediction_headers = prediction_reader.fieldnames or []

		if reference_headers != prediction_headers:
			raise ValueError(
				f"{reference_filename} and {prediction_filename} must have the same columns to compare reconstructed output"
			)

		value_columns = [header for header in reference_headers if header.lower() not in metadata_columns]
		if not value_columns:
			raise ValueError('No comparable value columns found in the reconstructed CSV files')

		reference_rows = {}
		for reference_row in reference_reader:
			key = tuple(reference_row[column] for column in key_columns)
			reference_rows[key] = reference_row

		prediction_rows = {}
		for prediction_row in prediction_reader:
			key = tuple(prediction_row[column] for column in key_columns)
			prediction_rows[key] = prediction_row

		common_keys = sorted(reference_rows.keys() & prediction_rows.keys())
		if not common_keys:
			raise ValueError('No matching rows found between the reconstructed CSV files')

		reference_sum = 0.0
		error_sum = 0.0
		sampled_rows = 0

		for key in common_keys:
			if rng.random() > sample_fraction:
				continue

			reference_row = reference_rows[key]
			prediction_row = prediction_rows[key]
			sampled_rows += 1

			for column in value_columns:
				reference_value = float(reference_row[column])
				prediction_value = float(prediction_row[column])
				if not (np.isfinite(reference_value) and np.isfinite(prediction_value)):
					continue
				reference_sum += reference_value * reference_value
				error_value = prediction_value - reference_value
				error_sum += error_value * error_value

		if sampled_rows == 0:
			raise ValueError('Sampling selected no rows; increase sample_fraction')

		if np.isclose(reference_sum, 0.0):
			raise ValueError('Reference reconstruction has zero norm')

	return float(np.sqrt(error_sum / reference_sum) * 100.0)


def oisst(n_columns : int = 25,
	device : str = None,
	standardize : bool = False,
	return_stats : bool = False) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
	'''
		Loads the OISST CSV data and converts it into one-step training pairs.

		The CSV contains a date column followed by 30 numeric columns. This function keeps the
		first n_columns numeric columns in sequential order, treats the full retained table as a
		single trajectory, and returns x/y pairs in the same way as duffing:
		x contains all but the final row and y contains all but the first row.

		Parameters
		------------------------------
		n_columns : int
			Number of sequential numeric columns to keep from the CSV. Must be between 1 and 30.

		device : str
			Device to send the returned tensors to. Defaults to CPU if not provided.

		standardize : bool
			If True, each retained PCA coefficient is centered and scaled to unit variance
			before building x/y pairs. This keeps the benchmark from being dominated by
			the leading EOF amplitudes.

		return_stats : bool
			If True, also returns the mean and scale tensors needed to map standardized
			coefficients back to physical EOF coefficient units.

		Returns
		------------------------------
		x, y[, stats]
			Output trajectory data as tensors. x[i] is the state at time i and y[i] is the state
			at the next time step.
	'''
	if n_columns <= 0:
		raise ValueError("n_columns must be positive")

	if n_columns > 30:
		raise ValueError("n_columns cannot exceed 30 for oisst.csv")

	if device is None:
		device = 'cpu'

	data_path = Path(__file__).resolve().parents[1] / 'data' / 'oisst.csv'
	data = np.genfromtxt(data_path, delimiter = ',', skip_header = 1, usecols = range(1, n_columns + 1))

	if data.ndim == 1:
		data = data[:, None]

	mean = np.mean(data, axis = 0)
	scale = np.std(data, axis = 0)
	scale[~np.isfinite(scale) | np.isclose(scale, 0.0)] = 1.0

	if standardize:
		data = (data - mean) / scale

	trajectory = torch.from_numpy(data).to(device = device, dtype = torch.float32)
	states = [trajectory]

	x = torch.cat([state[:-1] for state in states], dim = 0).to(device = device, dtype = trajectory.dtype)
	y = torch.cat([state[1:] for state in states], dim = 0).to(device = device, dtype = trajectory.dtype)

	if return_stats:
		stats = {
			'mean': torch.from_numpy(mean).to(device = device, dtype = trajectory.dtype),
			'scale': torch.from_numpy(scale).to(device = device, dtype = trajectory.dtype),
			'standardized': standardize,
		}
		return x, y, stats

	return x, y

def oisst_delay(n_columns : int = 30,
	n_lags : int = 12,
	device : str = None,
	standardize : bool = True,
	return_stats : bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, dict[str, torch.Tensor]]:
	'''
		Loads OISST PCA coefficients as one-step delay-coordinate training pairs.

		Each state is ordered as [PC_t, PC_{t-1}, ..., PC_{t-n_lags+1}], and the
		target is the one-month shifted delay state. The returned observable_x is the
		physical-unit current PC_t block, useful for plotting Koopman modes on EOF maps.
	'''
	if n_lags <= 0:
		raise ValueError("n_lags must be positive")

	x_one_step, y_one_step, stats = oisst(
		n_columns = n_columns,
		device = device,
		standardize = standardize,
		return_stats = True,
	)
	trajectory = torch.cat([x_one_step[:1], y_one_step], dim = 0)
	physical_trajectory = _to_physical_oisst_coefficients(trajectory, stats)
	if n_lags >= trajectory.shape[0]:
		raise ValueError("n_lags must be smaller than the number of OISST snapshots")

	x_states = []
	y_states = []
	observable_x = []
	for time_index in range(n_lags - 1, trajectory.shape[0] - 1):
		x_states.append(torch.cat([trajectory[time_index - lag] for lag in range(n_lags)], dim = 0))
		y_states.append(torch.cat([trajectory[time_index + 1 - lag] for lag in range(n_lags)], dim = 0))
		observable_x.append(physical_trajectory[time_index])

	x = torch.stack(x_states, dim = 0)
	y = torch.stack(y_states, dim = 0)
	observable_x = torch.stack(observable_x, dim = 0)
	predicted_dates = oisst_dates()[n_lags:]

	stats = {
		**stats,
		'n_lags': n_lags,
		'n_columns': n_columns,
		'state_dim': n_columns * n_lags,
	}

	if return_stats:
		return x, y, observable_x, predicted_dates, stats
	return x, y, observable_x, predicted_dates

def oisst_dates() -> np.ndarray:
	'''
		Loads the date column from oisst.csv.
	'''
	data_path = Path(__file__).resolve().parents[1] / 'data' / 'oisst.csv'
	dates = np.genfromtxt(data_path, delimiter = ',', skip_header = 1, usecols = 0, dtype = str)
	if dates.ndim == 0:
		dates = np.array([dates])
	return dates

def _to_physical_oisst_coefficients(coefficients : torch.Tensor, stats : dict[str, torch.Tensor] | None = None) -> torch.Tensor:
	'''
		Maps standardized OISST PCA coefficients back to physical EOF coefficient units.
	'''
	if stats is None or not stats.get('standardized', False):
		return coefficients
	return coefficients * stats['scale'].to(device = coefficients.device, dtype = coefficients.dtype) + stats['mean'].to(device = coefficients.device, dtype = coefficients.dtype)

def reconstruct_oisst_predictions(dictionary : TrainableDictionary, x : torch.Tensor, y : torch.Tensor,
	stats : dict[str, torch.Tensor] | None = None,
	output_columns : int | None = None,
) -> torch.Tensor:
	'''
		Reconstructs SST predictions from the learned lifted dynamics.

		The reconstruction uses the explicit formula
			Psi_Y \approx Psi_X K,
			C = argmin_C ||Psi_X C - Y||_F^2 = Psi_X^\dagger Y,
			\hat{Y} = Psi_X K C,
		where Psi_X = dictionary.evaluate(x) and Y = y.
	'''
	Psi_X = dictionary.evaluate(x)
	Psi_Y = dictionary.evaluate(y)
	W = torch.ones(x.shape[0], device = x.device, dtype = x.dtype) / x.shape[0]
	K = EDMD(Psi_X, Psi_Y, W)
	decoder = torch.linalg.lstsq(Psi_X, y).solution
	predictions = Psi_X @ K @ decoder
	if output_columns is not None:
		predictions = predictions[:, :output_columns]
	return _to_physical_oisst_coefficients(predictions, stats)

def save_oisst_predictions(predictions : torch.Tensor, dates : np.ndarray, dims : int, filename : str = 'predicted.csv') -> Path:
	'''
		Saves SST predictions to pyresdmd/data/filename.
	'''
	if dates.shape[0] != predictions.shape[0]:
		raise ValueError(f"dates has length {dates.shape[0]} but predictions has {predictions.shape[0]} rows")

	data_dir = Path(__file__).resolve().parents[1] / 'data'
	output_path = data_dir / filename
	column_names = ['time', *[f'sst_{i + 1}' for i in range(dims)]]
	array = predictions.detach().cpu().numpy()
	with open(output_path, 'w', newline = '') as file:
		writer = csv.writer(file)
		writer.writerow(column_names)
		for date, row in zip(dates, array):
			writer.writerow([date, *row.tolist()])
	return output_path

def reconstruct_oisst_signal(predicted_filename : str = 'predicted.csv', eofs_filename : str = 'eofs.csv', output_filename : str = 'reconstructed_signal.csv') -> Path:
	'''
		Reconstructs the OISST signal from predicted mode coefficients and EOF loadings.

		The reconstruction uses the shared leading modes in both files and writes a long-form
		CSV with columns time, lat, lon, and reconstructed_signal.
	'''
	data_dir = Path(__file__).resolve().parents[1] / 'data'
	predicted_path = data_dir / predicted_filename
	eofs_path = data_dir / eofs_filename
	output_path = data_dir / output_filename

	predicted_table = np.genfromtxt(predicted_path, delimiter = ',', skip_header = 1, dtype = str)
	if predicted_table.ndim == 1:
		predicted_table = predicted_table[None, :]
	if predicted_table.shape[1] < 2:
		raise ValueError(f"{predicted_filename} must contain a time column and at least one predicted mode")

	times = predicted_table[:, 0]
	predicted_modes = predicted_table[:, 1:].astype(float)

	eofs_table = np.genfromtxt(eofs_path, delimiter = ',', skip_header = 1)
	if eofs_table.ndim == 1:
		eofs_table = eofs_table[None, :]
	if eofs_table.shape[1] < 4:
		raise ValueError(f"{eofs_filename} must contain an index, lat, lon, and at least one EOF mode")

	lat_lon = eofs_table[:, 1:3]
	eof_modes = eofs_table[:, 3:]

	shared_modes = min(predicted_modes.shape[1], eof_modes.shape[1])
	if shared_modes == 0:
		raise ValueError('No shared EOF modes found between predicted.csv and eofs.csv')

	reconstructed = predicted_modes[:, :shared_modes] @ eof_modes[:, :shared_modes].T

	with open(output_path, 'w', newline = '') as file:
		writer = csv.writer(file)
		writer.writerow(['time', 'lat', 'lon', 'reconstructed_signal'])
		for time_value, row in zip(times, reconstructed):
			for (lat, lon), signal_value in zip(lat_lon, row):
				writer.writerow([time_value, float(lat), float(lon), float(signal_value)])

	return output_path

def oist_demo(dims, dictionary_size, hidden_layers = 4, hidden_dim = 128, save_plots = True, save_report = True, repeats = 15, device = None, epochs = 200, patience = 50, placeholder = False, poly_deg = 2, save_predictions = True, seed : int | None = 0, standardize : bool = True, batch_size : float = 1.0, use_residual_penalty : bool = True, residual_penalty_power : float = 1.0):
	t = time.time()
	if not device:
		device = 'cpu'
	l2_sample_fraction = 0.2
	_set_random_seed(seed)

	l_test_losses = []
	l_test_cond_nums = []
	l_test_forecast_errors = []
	l_test_l2_errors = []

	poly_loss = None
	poly_cond_num = None
	poly_forecast_error = None
	poly_l2_error = None
	saved_files = []

	x, y, oisst_stats = oisst(n_columns = dims, device = device, standardize = standardize, return_stats = True)
	x_observable = _to_physical_oisst_coefficients(x, oisst_stats)
	predicted_dates = oisst_dates()[1:]

	sampled_baseline = None
	if save_predictions:
		print('Loading sampled baseline reconstruction...')
		sampled_baseline = load_sampled_baseline(dates = predicted_dates, sample_fraction = l2_sample_fraction, seed = 0)
		print(f"Sampled {sampled_baseline['baseline_values'].shape[0]} baseline points")

	for i in range(repeats):
		print(f"Repeat {i + 1}/{repeats} for ReLUDictionary")
		_set_random_seed(None if seed is None else seed + i)
		base = ReLUDictionary(input_dim = dims, n_functions = dictionary_size, hidden_layers = hidden_layers, hidden_dim = hidden_dim) # was 4, 128
		dictionary = TrainableDictionary(base)
		dictionary.to(device)

		if i == 0 and save_plots:
			report = dictionary.report(x, y)
			init_eigvals = report['eigenvalues']
			init_eigvecs = report['eigenvectors']
			init_residuals = report['residuals']
			Psi_X, Psi_Y = dictionary.evaluate(x), dictionary.evaluate(y)
			init_pseudospec = compute_pseudospectra(Psi_X, Psi_Y)
			init_modes = compute_koopman_modes(Psi_X, x_observable, init_eigvecs)

		trained = dictionary.fit(x, y, epochs = epochs, patience = patience, shuffle = False, batch_size = batch_size)
		train_losses = trained['train_losses']
		test_losses = trained['test_losses']
		train_cond_nums = trained['train_cond_nums']
		test_cond_nums = trained['test_cond_nums']
		train_forecast_errors = trained['train_forecast_errors']
		test_forecast_errors = trained['test_forecast_errors']
		final_epoch = trained['final_epoch']

		best_model = np.argmin(np.array(test_losses.cpu().numpy() if isinstance(test_losses, torch.Tensor) else test_losses))
		best_loss = test_losses[best_model]
		best_cond_num = test_cond_nums[best_model]
		best_forecast_error = test_forecast_errors[best_model]

		predicted_sst = None
		with torch.no_grad():
			if i == 0 and save_plots:
				Psi_X, Psi_Y = dictionary.evaluate(x), dictionary.evaluate(y)
				final_pseudospec = compute_pseudospectra(Psi_X, Psi_Y)
				W = torch.ones(x.shape[0], device = x.device) / x.shape[0]
				final_eigvals, final_eigvecs = compute_eigendecomposition_from_weights(Psi_X, Psi_Y, W)
				final_residuals = compute_residuals(final_eigvals, final_eigvecs, Psi_X, Psi_Y, W)
				final_modes = compute_koopman_modes(Psi_X, x_observable, final_eigvecs)
			if sampled_baseline is not None:
				predicted_sst = reconstruct_oisst_predictions(dictionary, x, y, oisst_stats)

		l_test_losses.append(best_loss.item() if isinstance(best_loss, torch.Tensor) else best_loss)
		l_test_cond_nums.append(best_cond_num.item() if isinstance(best_cond_num, torch.Tensor) else best_cond_num)
		l_test_forecast_errors.append(best_forecast_error.item() if isinstance(best_forecast_error, torch.Tensor) else best_forecast_error)

		if sampled_baseline is not None and predicted_sst is not None:
			predicted_modes = predicted_sst.detach().cpu().numpy()
			l_test_l2_errors.append(reconstruction_relative_l2_error_from_modes(predicted_modes, sampled_baseline))
			if i == 0:
				prediction_path = save_oisst_predictions(predicted_sst, predicted_dates, dims)
				saved_files.append(prediction_path.name)

		poly_metrics = None
		if placeholder and i == 0:
			h_dict = PolynomialDictionary(input_dim = dims, degree = poly_deg)
			h_dict.to(device)
			poly_metrics = benchmark_metrics(h_dict, x, y)
			poly_loss = poly_metrics['loss'].item() if isinstance(poly_metrics['loss'], torch.Tensor) else poly_metrics['loss']
			poly_cond_num = poly_metrics['cond_num'].item() if isinstance(poly_metrics['cond_num'], torch.Tensor) else poly_metrics['cond_num']
			poly_forecast_error = poly_metrics['forecast_error'].item() if isinstance(poly_metrics['forecast_error'], torch.Tensor) else poly_metrics['forecast_error']

			if sampled_baseline is not None:
				with torch.no_grad():
					poly_predicted_sst = reconstruct_oisst_predictions(h_dict, x, y, oisst_stats)
				poly_predicted_modes = poly_predicted_sst.detach().cpu().numpy()
				poly_l2_error = reconstruction_relative_l2_error_from_modes(poly_predicted_modes, sampled_baseline)

			if save_plots:
				with torch.no_grad():
					poly_Psi_X, poly_Psi_Y = h_dict.evaluate(x), h_dict.evaluate(y)
					poly_W = torch.ones(x.shape[0], device = x.device, dtype = x.dtype) / x.shape[0]
					poly_eigvals, poly_eigvecs = compute_eigendecomposition_from_weights(poly_Psi_X, poly_Psi_Y, poly_W)
					poly_residuals = compute_residuals(poly_eigvals, poly_eigvecs, poly_Psi_X, poly_Psi_Y, poly_W)
					poly_modes = compute_koopman_modes(poly_Psi_X, x_observable, poly_eigvecs)

				if 'final_modes' in locals() and 'init_modes' in locals():
					f_el_nino_mode_comparison = f'oisst_el_nino_mode_comparison_r{dims}_{int(time.time())}.png'
					el_nino_plot = plot_oisst_el_nino_mode_comparison(
						final_modes['modes'],
						init_modes['modes'],
						poly_modes['modes'],
						trained_residuals = final_residuals,
						init_residuals = init_residuals,
						poly_residuals = poly_residuals,
						filename = f_el_nino_mode_comparison,
						save = True,
						show = False,
						use_residual_penalty = use_residual_penalty,
						residual_penalty_power = residual_penalty_power,
					)
					saved_files.append(Path(el_nino_plot['filename']).name)
					print(
						"El Nino mode comparison: "
						f"trained={el_nino_plot['trained_mode_index']} "
						f"(Nino3.4 agreement={el_nino_plot['trained_nino34_agreement']:.3f}), "
						f"untrained={el_nino_plot['init_mode_index']} "
						f"(agreement={el_nino_plot['init_agreement']:.3f}), "
						f"polynomial={el_nino_plot['poly_mode_index']} "
						f"(agreement={el_nino_plot['poly_agreement']:.3f})"
					)

		if i == 0 and save_plots:
			f1 = f'oisst_sim_loss_{int(time.time())}.png'
			saved_files.append(f1)
			plot_curve(final_epoch, train_losses, test_losses, add_values = {'Polynomial': poly_metrics['loss']} if poly_metrics is not None else {},
				   displayname = 'OISST', ylabel = 'Loss', save_plot = True,
				   filename = f1)

			f2 = f'oisst_sim_cond_num_{int(time.time())}.png'
			saved_files.append(f2)
			plot_curve(final_epoch, train_cond_nums, test_cond_nums, add_values = {},
					displayname = 'OISST', ylabel = 'Condition number', save_plot = True,
					filename = f2)

			f3 = f'oisst_sim_forecast_{int(time.time())}.png'
			saved_files.append(f3)
			plot_curve(final_epoch, train_forecast_errors, test_forecast_errors, add_values = {'Polynomial': poly_metrics['forecast_error']} if poly_metrics is not None else {},
					displayname = 'OISST', ylabel = 'Forecast error', save_plot = True,
					filename = f3)

			eps_levels = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]
			re_vals = init_pseudospec['re_vals']
			im_vals = init_pseudospec['im_vals']
			tau_grid = init_pseudospec['tau_grid']
			f4 = f'oisst_pseudospec_init_relu_{int(time.time())}.png'
			saved_files.append(f4)
			plot_pseudospec(re_vals, im_vals, tau_grid, eps_levels, eigvals = init_eigvals, unit_circle = True, save = True, displayname = 'Psuedospectrum of OISST (Untrained ReLU)', filename = f4)
			re_vals = final_pseudospec['re_vals']
			im_vals = final_pseudospec['im_vals']
			tau_grid = final_pseudospec['tau_grid']
			f5 = f'oisst_pseudospec_final_relu_{int(time.time())}.png'
			saved_files.append(f5)
			plot_pseudospec(re_vals, im_vals, tau_grid, eps_levels, eigvals = final_eigvals, unit_circle = True, save = True, displayname = 'Psuedospectrum of OISST (Trained ReLU)', filename = f5)
			if poly_metrics is not None:
				re_vals = poly_metrics['pseudospec']['re_vals']
				im_vals = poly_metrics['pseudospec']['im_vals']
				tau_grid = poly_metrics['pseudospec']['tau_grid']
				f6 = f'oisst_pseudospec_polynomial_{int(time.time())}.png'
				saved_files.append(f6)
				plot_pseudospec(re_vals, im_vals, tau_grid, eps_levels, eigvals = poly_metrics['Lambda'], unit_circle = True, save = True, displayname = 'Psuedospectrum of OISST (Polynomial)', filename = f6)

	reconstruction_l2_summary = 'n/a'
	if l_test_l2_errors:
		reconstruction_l2_summary = f"{np.mean(l_test_l2_errors):.4f} +- {np.std(l_test_l2_errors):.4f} reconstruction L2 error"

	print(f"ReLUDictionary: {np.mean(l_test_losses):.4f} +- {np.std(l_test_losses):.4f} loss, {np.mean(l_test_cond_nums):.4f} +- {np.std(l_test_cond_nums):.4f} cond num, {np.mean(l_test_forecast_errors):.4f} +- {np.std(l_test_forecast_errors):.4f} forecast error, {reconstruction_l2_summary}")

	if placeholder:
		poly_reconstruction_l2_summary = 'n/a'
		if poly_l2_error is not None:
			poly_reconstruction_l2_summary = f"{poly_l2_error:.4f} reconstruction L2 error"
		print(f"PolynomialDictionary: {poly_loss:.4f} loss, {poly_cond_num:.4f} cond num, {poly_forecast_error:.4f} forecast error, {poly_reconstruction_l2_summary}")

	if save_plots and saved_files:
		print("Saved files:")
		for filename in saved_files:
			print(f"  {filename}")

	if save_report:
		reconstruction_l2_line = '\t\t\tMean reconstruction L2 error for ReLU: n/a'
		if l_test_l2_errors:
			reconstruction_l2_line = f"\t\t\tMean reconstruction L2 error for ReLU: {np.mean(l_test_l2_errors)} (std: {np.std(l_test_l2_errors)})"
		report = f"""
		Ran OISST simulation {repeats} times.
		----------------------------------------------------------------------------
			Mean best loss for ReLU: {np.mean(l_test_losses)} (std: {np.std(l_test_losses)})
			Mean condition number for best ReLU: {np.mean(l_test_cond_nums)} (std: {np.std(l_test_cond_nums)})
			Mean forecast error for best ReLU: {np.mean(l_test_forecast_errors)} (std: {np.std(l_test_forecast_errors)})
			{reconstruction_l2_line}
			"""
		if placeholder:
			poly_reconstruction_l2_line = '\t\t\t\tReconstruction L2 error for Polynomial: n/a'
			if poly_l2_error is not None:
				poly_reconstruction_l2_line = f"\t\t\t\tReconstruction L2 error for Polynomial: {poly_l2_error}"
			report += f"""
			---------------------------------------------------------------------------------
				Loss for Polynomial: {poly_loss}
				Condition number for Polynomial: {poly_cond_num}
				Forecast error for Polynomial: {poly_forecast_error}
{poly_reconstruction_l2_line}
			"""
		if save_plots and saved_files:
			report += """
			---------------------------------------------------------------------------------
			Saved files:
			"""
			report += "\n" + "\n".join(f"                {filename}" for filename in saved_files)
		with open(f"log_{int(time.time())}.log", "x") as file:
			file.write(report)


def oist_delay_hybrid_demo(dims : int = 30,
	dictionary_size : int = 60,
	n_lags : int = 6,
	save_plots : bool = True,
	save_report : bool = True,
	repeats : int = 3,
	device = None,
	epochs : int = 200,
	patience : int = 50,
	compare_polynomial : bool = True,
	poly_deg : int = 1,
	save_predictions : bool = True,
	seed : int | None = 0,
	standardize : bool = True,
	batch_size : float = 1.0,
	lr : float = 1e-4,
	weight_decay : float = 1e-3,
	test_size : float = 0.5,
	hidden_dim : int = 32,
	hidden_layers : int = 2,
	use_residual_penalty : bool = True,
	residual_penalty_power : float = 1.0,
    shuffle : bool = False,
    include_constant : bool = False,
    include_identity : bool = False
):
	'''
		Runs the OISST benchmark with delay coordinates and a hybrid dictionary.
	'''
	if not device:
		device = 'cpu'
	l2_sample_fraction = 0.2
	_set_random_seed(seed)

	l_test_losses = []
	l_test_cond_nums = []
	l_test_forecast_errors = []
	l_test_l2_errors = []
	poly_loss = None
	poly_cond_num = None
	poly_forecast_error = None
	poly_l2_error = None
	poly_eigvals = None
	poly_residuals = None
	poly_modes = None
	saved_files = []

	x, y, x_observable, predicted_dates, oisst_stats = oisst_delay(
		n_columns = dims,
		n_lags = n_lags,
		device = device,
		standardize = standardize,
		return_stats = True,
	)
	feature_count = 1 + x.shape[1] + dictionary_size
	n_train = int((1 - test_size) * x.shape[0])
	n_test = x.shape[0] - n_train
	if feature_count > min(n_train, n_test):
		raise ValueError(
			"Hybrid dictionary has more observables than one side of the train/test split: "
			f"{feature_count} observables, {n_train} train samples, {n_test} test samples. "
			"Reduce dims, n_lags, or dictionary_size, or choose a more balanced/larger split."
		)

	sampled_baseline = None
	if save_predictions:
		print('Loading sampled baseline reconstruction...')
		sampled_baseline = load_sampled_baseline(dates = predicted_dates, sample_fraction = l2_sample_fraction, seed = 0)
		print(f"Sampled {sampled_baseline['baseline_values'].shape[0]} baseline points")

	poly_dictionary = None
	poly_size = None
	if compare_polynomial:
		poly_dictionary = PolynomialDictionary(input_dim = x.shape[1], degree = poly_deg)
		poly_dictionary.to(device)
		poly_size = poly_dictionary.size
		if poly_size > x.shape[0]:
			print(
				"Skipping polynomial comparison: "
				f"degree {poly_deg} on delayed dimension {x.shape[1]} creates {poly_size} observables "
				f"for only {x.shape[0]} samples."
			)
			poly_dictionary = None
		else:
			print(f"Computing PolynomialDictionary degree {poly_deg} baseline ({poly_size} observables)")
			with torch.no_grad():
				poly_reporter = TrainableDictionary(poly_dictionary)
				poly_reporter.to(device)
				poly_report = poly_reporter.report(x, y)
				poly_loss = poly_report['loss'].item() if isinstance(poly_report['loss'], torch.Tensor) else poly_report['loss']
				poly_cond_num = poly_report['cond_num'].item() if isinstance(poly_report['cond_num'], torch.Tensor) else poly_report['cond_num']
				poly_forecast_error = poly_report['forecast_error'].item() if isinstance(poly_report['forecast_error'], torch.Tensor) else poly_report['forecast_error']
				poly_eigvals = poly_report['eigenvalues']
				poly_eigvecs = poly_report['eigenvectors']
				poly_residuals = poly_report['residuals']
				poly_Psi_X = poly_dictionary.evaluate(x)
				poly_modes = compute_koopman_modes(poly_Psi_X, x_observable, poly_eigvecs)
				if sampled_baseline is not None:
					poly_predicted_sst = reconstruct_oisst_predictions(
						poly_dictionary,
						x,
						y,
						oisst_stats,
						output_columns = dims,
					)
					poly_l2_error = reconstruction_relative_l2_error_from_modes(
						poly_predicted_sst.detach().cpu().numpy(),
						sampled_baseline,
					)

	final_el_nino_summary = None

	for i in range(repeats):
		print(f"Repeat {i + 1}/{repeats} for HybridFixedReLUDictionary")
		_set_random_seed(None if seed is None else seed + i)
		base = HybridFixedReLUDictionary(
			input_dim = x.shape[1],
			n_functions = dictionary_size,
            include_constant = include_constant,
            include_identity = include_identity,
			hidden_layers = hidden_layers,
			hidden_dim = hidden_dim,
		)
		dictionary = TrainableDictionary(base)
		dictionary.to(device)

		init_eigvals = None
		init_residuals = None
		init_modes = None
		if i == 0 and save_plots:
			with torch.no_grad():
				init_report = dictionary.report(x, y)
				init_eigvals = init_report['eigenvalues']
				init_eigvecs = init_report['eigenvectors']
				init_residuals = init_report['residuals']
				init_Psi_X = dictionary.evaluate(x)
				init_modes = compute_koopman_modes(init_Psi_X, x_observable, init_eigvecs)

		trained = dictionary.fit(
			x,
			y,
			epochs = epochs,
			patience = patience,
			shuffle = shuffle,
			batch_size = batch_size,
			lr = lr,
			weight_decay = weight_decay,
			test_size = test_size,
		)
		train_losses = trained['train_losses']
		test_losses = trained['test_losses']
		train_cond_nums = trained['train_cond_nums']
		test_cond_nums = trained['test_cond_nums']
		train_forecast_errors = trained['train_forecast_errors']
		test_forecast_errors = trained['test_forecast_errors']
		final_epoch = trained['final_epoch']

		best_model = np.argmin(np.array(test_losses.cpu().numpy() if isinstance(test_losses, torch.Tensor) else test_losses))
		best_loss = test_losses[best_model]
		best_cond_num = test_cond_nums[best_model]
		best_forecast_error = test_forecast_errors[best_model]

		predicted_sst = None
		with torch.no_grad():
			Psi_X, Psi_Y = dictionary.evaluate(x), dictionary.evaluate(y)
			W = torch.ones(x.shape[0], device = x.device, dtype = x.dtype) / x.shape[0]
			final_eigvals, final_eigvecs = compute_eigendecomposition_from_weights(Psi_X, Psi_Y, W)
			final_residuals = compute_residuals(final_eigvals, final_eigvecs, Psi_X, Psi_Y, W)
			final_modes = compute_koopman_modes(Psi_X, x_observable, final_eigvecs)
			final_el_nino_summary = locate_oisst_el_nino_mode(
				final_modes['modes'],
				residuals = final_residuals,
				use_residual_penalty = use_residual_penalty,
				residual_penalty_power = residual_penalty_power,
			)
			if sampled_baseline is not None:
				predicted_sst = reconstruct_oisst_predictions(
					dictionary,
					x,
					y,
					oisst_stats,
					output_columns = dims,
				)

		l_test_losses.append(best_loss.item() if isinstance(best_loss, torch.Tensor) else best_loss)
		l_test_cond_nums.append(best_cond_num.item() if isinstance(best_cond_num, torch.Tensor) else best_cond_num)
		l_test_forecast_errors.append(best_forecast_error.item() if isinstance(best_forecast_error, torch.Tensor) else best_forecast_error)

		if sampled_baseline is not None and predicted_sst is not None:
			predicted_modes = predicted_sst.detach().cpu().numpy()
			l_test_l2_errors.append(reconstruction_relative_l2_error_from_modes(predicted_modes, sampled_baseline))
			if i == 0:
				prediction_path = save_oisst_predictions(predicted_sst, predicted_dates, dims)
				saved_files.append(prediction_path.name)

		if i == 0 and save_plots:
			f1 = f'oisst_delay_hybrid_loss_r{dims}_lag{n_lags}_{int(time.time())}.png'
			saved_files.append(f1)
			plot_curve(final_epoch, train_losses, test_losses, add_values = {'Polynomial': poly_loss} if poly_loss is not None else {},
				   displayname = f'OISST delay hybrid ({n_lags} lags)', ylabel = 'Loss', save_plot = True,
				   filename = f1)

			f2 = f'oisst_delay_hybrid_cond_num_r{dims}_lag{n_lags}_{int(time.time())}.png'
			saved_files.append(f2)
			plot_curve(final_epoch, train_cond_nums, test_cond_nums, add_values = {},
					displayname = f'OISST delay hybrid ({n_lags} lags)', ylabel = 'Condition number', save_plot = True,
					filename = f2)

			f3 = f'oisst_delay_hybrid_forecast_r{dims}_lag{n_lags}_{int(time.time())}.png'
			saved_files.append(f3)
			plot_curve(final_epoch, train_forecast_errors, test_forecast_errors, add_values = {'Polynomial': poly_forecast_error} if poly_forecast_error is not None else {},
					displayname = f'OISST delay hybrid ({n_lags} lags)', ylabel = 'Forecast error', save_plot = True,
					filename = f3)

			mode_index = final_el_nino_summary['mode_index']
			mode_residual = final_el_nino_summary['residual']
			mode_agreement = final_el_nino_summary['agreement']
			f4 = f'oisst_delay_hybrid_el_nino_mode_r{dims}_lag{n_lags}_{int(time.time())}.png'
			saved_files.append(f4)
			plot_oisst_koopman_mode(
				final_modes['modes'],
				mode_index,
				eigenvalues = final_eigvals,
				component = 'real',
				filename = f4,
				title = f"OISST delay hybrid El Nino mode {mode_index}, residual={mode_residual:.3g}, Nino3.4={mode_agreement:.3f}",
				save = True,
				show = False,
			)

			if init_modes is not None and poly_modes is not None:
				f5 = f'oisst_delay_hybrid_el_nino_comparison_r{dims}_lag{n_lags}_{int(time.time())}.png'
				el_nino_plot = plot_oisst_el_nino_mode_comparison(
					final_modes['modes'],
					init_modes['modes'],
					poly_modes['modes'],
					trained_residuals = final_residuals,
					init_residuals = init_residuals,
					poly_residuals = poly_residuals,
					filename = f5,
					save = True,
					show = False,
					use_residual_penalty = use_residual_penalty,
					residual_penalty_power = residual_penalty_power,
				)
				saved_files.append(Path(el_nino_plot['filename']).name)
				print(
					"El Nino mode comparison: "
					f"trained={el_nino_plot['trained_mode_index']} "
					f"(Nino3.4 agreement={el_nino_plot['trained_nino34_agreement']:.3f}), "
					f"untrained={el_nino_plot['init_mode_index']} "
					f"(agreement={el_nino_plot['init_agreement']:.3f}), "
					f"polynomial={el_nino_plot['poly_mode_index']} "
					f"(agreement={el_nino_plot['poly_agreement']:.3f})"
				)

			mode_agreement_limit = min(40, final_modes['modes'].shape[0], init_modes['modes'].shape[0]) if init_modes is not None else 0
			if init_modes is not None and init_residuals is not None and mode_index < mode_agreement_limit:
				f6 = f'oisst_delay_hybrid_mode_agreement_r{dims}_lag{n_lags}_{int(time.time())}.png'
				agreement_plot = plot_oisst_mode_agreement(
					final_modes['modes'],
					init_modes['modes'],
					reference_name = 'Trained hybrid',
					comparison_name = 'Initial hybrid',
					reference_mode_index = mode_index,
					filename = f6,
					save = True,
					show = False,
					max_modes = mode_agreement_limit,
				)
				saved_files.append(Path(agreement_plot['filename']).name)

		print(
			"Delay hybrid El Nino mode: "
			f"index={final_el_nino_summary['mode_index']}, "
			f"residual={final_el_nino_summary['residual']:.4f}, "
			f"Nino3.4 agreement={final_el_nino_summary['agreement']:.3f}, "
			f"score={final_el_nino_summary['score']:.3f}, "
			f"residual_penalty={final_el_nino_summary['used_residual_penalty']}"
		)

	reconstruction_l2_summary = 'n/a'
	if l_test_l2_errors:
		reconstruction_l2_summary = f"{np.mean(l_test_l2_errors):.4f} +- {np.std(l_test_l2_errors):.4f} reconstruction L2 error"

	print(
		f"HybridFixedReLUDictionary ({n_lags} lags): "
		f"{np.mean(l_test_losses):.4f} +- {np.std(l_test_losses):.4f} loss, "
		f"{np.mean(l_test_cond_nums):.4f} +- {np.std(l_test_cond_nums):.4f} cond num, "
		f"{np.mean(l_test_forecast_errors):.4f} +- {np.std(l_test_forecast_errors):.4f} forecast error, "
		f"{reconstruction_l2_summary}"
	)
	if poly_loss is not None:
		poly_reconstruction_l2_summary = 'n/a'
		if poly_l2_error is not None:
			poly_reconstruction_l2_summary = f"{poly_l2_error:.4f} reconstruction L2 error"
		print(
			f"PolynomialDictionary degree {poly_deg} ({n_lags} lags): "
			f"{poly_loss:.4f} loss, {poly_cond_num:.4f} cond num, "
			f"{poly_forecast_error:.4f} forecast error, {poly_reconstruction_l2_summary}"
		)

	if save_plots and saved_files:
		print("Saved files:")
		for filename in saved_files:
			print(f"  {filename}")

	if save_report:
		reconstruction_l2_line = '\t\t\tMean reconstruction L2 error for hybrid delay ReLU: n/a'
		if l_test_l2_errors:
			reconstruction_l2_line = f"\t\t\tMean reconstruction L2 error for hybrid delay ReLU: {np.mean(l_test_l2_errors)} (std: {np.std(l_test_l2_errors)})"
		el_nino_line = '\t\t\tFinal El Nino mode: n/a'
		if final_el_nino_summary is not None:
			el_nino_line = (
				f"\t\t\tFinal El Nino mode: {final_el_nino_summary['mode_index']} "
				f"(residual: {final_el_nino_summary['residual']}, "
				f"Nino3.4 agreement: {final_el_nino_summary['agreement']}, "
				f"score: {final_el_nino_summary['score']}, "
				f"residual penalty: {final_el_nino_summary['used_residual_penalty']})"
			)
		poly_report_line = '\t\t\tPolynomial baseline: n/a'
		if poly_loss is not None:
			poly_report_line = (
				f"\t\t\tPolynomial baseline degree {poly_deg}: loss {poly_loss}, "
				f"condition number {poly_cond_num}, forecast error {poly_forecast_error}"
			)
			if poly_l2_error is not None:
				poly_report_line += f", reconstruction L2 error {poly_l2_error}"
		report = f"""
		Ran OISST delay hybrid simulation {repeats} times.
		----------------------------------------------------------------------------
			PC dimensions: {dims}
			Delay lags: {n_lags}
			Hybrid dictionary size: {1 + x.shape[1] + dictionary_size} ({x.shape[1]} identity, 1 constant, {dictionary_size} trainable)
			Mean best loss for hybrid ReLU: {np.mean(l_test_losses)} (std: {np.std(l_test_losses)})
			Mean condition number for best hybrid ReLU: {np.mean(l_test_cond_nums)} (std: {np.std(l_test_cond_nums)})
			Mean forecast error for best hybrid ReLU: {np.mean(l_test_forecast_errors)} (std: {np.std(l_test_forecast_errors)})
			{reconstruction_l2_line}
			{el_nino_line}
			{poly_report_line}
			"""
		if save_plots and saved_files:
			report += """
			---------------------------------------------------------------------------------
			Saved files:
			"""
			report += "\n" + "\n".join(f"                {filename}" for filename in saved_files)
		with open(f"log_delay_hybrid_{int(time.time())}.log", "x") as file:
			file.write(report)


def old_oist_demo(dims, dictionary_size, save_plots = True, save_report = True, repeats = 15, device = None, epochs = 200, patience = 50, placeholder = False, poly_deg = 2, save_predictions = True):
	t = time.time()
	if not device:
		device = 'cpu'
	l2_sample_fraction = 0.05

	l_test_losses = []
	l_test_cond_nums = []
	l_test_forecast_errors = []
	l_test_l2_errors = []

	l_poly_losses = []
	l_poly_cond_nums = []
	l_poly_forecast_errors = []
	saved_files = []

	for i in range(repeats):
		print(f"Repeat {i + 1}/{repeats} for ReLUDictionary")
		base = ReLUDictionary(input_dim = dims, n_functions = dictionary_size, hidden_layers = 3, hidden_dim = 96) # was 4, 128
		dictionary = TrainableDictionary(base)
		dictionary.to(device)

		x, y = oisst(n_columns = dims, device = device)

		report = dictionary.report(x, y)
		init_eigvals = report['eigenvalues']
		Psi_X, Psi_Y = dictionary.evaluate(x), dictionary.evaluate(y)
		init_pseudospec = compute_pseudospectra(Psi_X, Psi_Y)

		trained = dictionary.fit(x, y, epochs = epochs, patience = patience, shuffle = False)
		train_losses = trained['train_losses']
		test_losses = trained['test_losses']
		train_cond_nums = trained['train_cond_nums']
		test_cond_nums = trained['test_cond_nums']
		train_forecast_errors = trained['train_forecast_errors']
		test_forecast_errors = trained['test_forecast_errors']
		final_epoch = trained['final_epoch']

		best_model = np.argmin(np.array(test_losses.cpu().numpy() if isinstance(test_losses, torch.Tensor) else test_losses))
		print(test_losses[-10:], best_model, test_losses[best_model])
		best_loss = test_losses[best_model]
		best_cond_num = test_cond_nums[best_model]
		best_forecast_error = test_forecast_errors[best_model]
		with torch.no_grad():
			Psi_X, Psi_Y = dictionary.evaluate(x), dictionary.evaluate(y)
			final_pseudospec = compute_pseudospectra(Psi_X, Psi_Y)
			W = torch.ones(x.shape[0], device = x.device) / x.shape[0]
			final_eigvals, _ = compute_eigendecomposition_from_weights(Psi_X, Psi_Y, W)
			predicted_sst = reconstruct_oisst_predictions(dictionary, x, y)
			predicted_dates = oisst_dates()[1:]

		l_test_losses.append(best_loss.item() if isinstance(best_loss, torch.Tensor) else best_loss)
		l_test_cond_nums.append(best_cond_num.item() if isinstance(best_cond_num, torch.Tensor) else best_cond_num)
		l_test_forecast_errors.append(best_forecast_error.item() if isinstance(best_forecast_error, torch.Tensor) else best_forecast_error)

		if save_predictions and predicted_sst is not None and predicted_dates is not None:
			prediction_path = save_oisst_predictions(predicted_sst, predicted_dates, dims)
			reconstructed_path = reconstruct_oisst_signal(prediction_path.name)
			l_test_l2_errors.append(
				reconstruction_relative_l2_error(
					'reconstructed_baseline.csv',
					reconstructed_path.name,
					sample_fraction = l2_sample_fraction,
					seed = 0,
				)
			)
			if i == 0:
				saved_files.append(prediction_path.name)
				saved_files.append(reconstructed_path.name)

		poly_metrics = None
		if placeholder:
			h_dict = PolynomialDictionary(input_dim = dims, degree = poly_deg)
			h_dict.to(device)
			poly_metrics = benchmark_metrics(h_dict, x, y)
			l_poly_losses.append(poly_metrics['loss'].item() if isinstance(poly_metrics['loss'], torch.Tensor) else poly_metrics['loss'])
			l_poly_cond_nums.append(poly_metrics['cond_num'].item() if isinstance(poly_metrics['cond_num'], torch.Tensor) else poly_metrics['cond_num'])
			l_poly_forecast_errors.append(poly_metrics['forecast_error'].item() if isinstance(poly_metrics['forecast_error'], torch.Tensor) else poly_metrics['forecast_error'])

		if i == 0 and save_plots:
			f1 = f'oisst_sim_loss_{int(time.time())}.png'
			saved_files.append(f1)
			plot_curve(final_epoch, train_losses, test_losses, add_values = {'Polynomial': poly_metrics['loss']} if poly_metrics is not None else {},
				   displayname = 'OISST', ylabel = 'Loss', save_plot = True,
				   filename = f1)

			f2 = f'oisst_sim_cond_num_{int(time.time())}.png'
			saved_files.append(f2)
			plot_curve(final_epoch, train_cond_nums, test_cond_nums, add_values = {'Polynomial': poly_metrics['cond_num']} if poly_metrics is not None else {},
					displayname = 'OISST', ylabel = 'Condition number', save_plot = True,
					filename = f2)

			f3 = f'oisst_sim_forecast_{int(time.time())}.png'
			saved_files.append(f3)
			plot_curve(final_epoch, train_forecast_errors, test_forecast_errors, add_values = {'Polynomial': poly_metrics['forecast_error']} if poly_metrics is not None else {},
					displayname = 'OISST', ylabel = 'Forecast error', save_plot = True,
					filename = f3)

		if i == 0 and save_plots:
			eps_levels = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]
			re_vals = init_pseudospec['re_vals']
			im_vals = init_pseudospec['im_vals']
			tau_grid = init_pseudospec['tau_grid']
			f4 = f'oisst_pseudospec_init_relu_{int(time.time())}.png'
			saved_files.append(f4)
			plot_pseudospec(re_vals, im_vals, tau_grid, eps_levels, eigvals = init_eigvals, unit_circle = True, save = True, displayname = 'Psuedospectrum of OISST (Untrained ReLU)', filename = f4)
			re_vals = final_pseudospec['re_vals']
			im_vals = final_pseudospec['im_vals']
			tau_grid = final_pseudospec['tau_grid']
			f5 = f'oisst_pseudospec_final_relu_{int(time.time())}.png'
			saved_files.append(f5)
			plot_pseudospec(re_vals, im_vals, tau_grid, eps_levels, eigvals = final_eigvals, unit_circle = True, save = True, displayname = 'Psuedospectrum of OISST (Trained ReLU)', filename = f5)
			if poly_metrics is not None:
				re_vals = poly_metrics['pseudospec']['re_vals']
				im_vals = poly_metrics['pseudospec']['im_vals']
				tau_grid = poly_metrics['pseudospec']['tau_grid']
				f6 = f'oisst_pseudospec_polynomial_{int(time.time())}.png'
				saved_files.append(f6)
				plot_pseudospec(re_vals, im_vals, tau_grid, eps_levels, eigvals = poly_metrics['Lambda'], unit_circle = True, save = True, displayname = 'Psuedospectrum of OISST (Polynomial)', filename = f6)

	reconstruction_l2_summary = 'n/a'
	if l_test_l2_errors:
		reconstruction_l2_summary = f"{np.mean(l_test_l2_errors):.4f} +- {np.std(l_test_l2_errors):.4f} reconstruction L2 error"

	print(f"ReLUDictionary: {np.mean(l_test_losses):.4f} +- {np.std(l_test_losses):.4f} loss, {np.mean(l_test_cond_nums):.4f} +- {np.std(l_test_cond_nums):.4f} cond num, {np.mean(l_test_forecast_errors):.4f} +- {np.std(l_test_forecast_errors):.4f} forecast error, {reconstruction_l2_summary}")

	if placeholder:
		print(f"PolynomialDictionary: {np.mean(l_poly_losses):.4f} +- {np.std(l_poly_losses):.4f} loss, {np.mean(l_poly_cond_nums):.4f} +- {np.std(l_poly_cond_nums):.4f} cond num, {np.mean(l_poly_forecast_errors):.4f} +- {np.std(l_poly_forecast_errors):.4f} forecast error")

	if save_plots and saved_files:
		print("Saved files:")
		for filename in saved_files:
			print(f"  {filename}")

	if save_report:
		reconstruction_l2_line = '\t\t\tMean reconstruction L2 error for ReLU: n/a'
		if l_test_l2_errors:
			reconstruction_l2_line = f"\t\t\tMean reconstruction L2 error for ReLU: {np.mean(l_test_l2_errors)} (std: {np.std(l_test_l2_errors)})"
		report = f"""
		Ran OISST simulation {repeats} times.
		----------------------------------------------------------------------------
			Mean best loss for ReLU: {np.mean(l_test_losses)} (std: {np.std(l_test_losses)})
			Mean condition number for best ReLU: {np.mean(l_test_cond_nums)} (std: {np.std(l_test_cond_nums)})
			Mean forecast error for best ReLU: {np.mean(l_test_forecast_errors)} (std: {np.std(l_test_forecast_errors)})
			{reconstruction_l2_line}
			"""
		if placeholder:
			report += f"""
			---------------------------------------------------------------------------------
				Mean loss for Polynomial: {np.mean(l_poly_losses)} (std: {np.std(l_poly_losses)})
				Mean condition number for Polynomial: {np.mean(l_poly_cond_nums)} (std: {np.std(l_poly_cond_nums)})
				Mean forecast error for Polynomial: {np.mean(l_poly_forecast_errors)} (std: {np.std(l_poly_forecast_errors)})
			"""
		if save_plots and saved_files:
			report += """
			---------------------------------------------------------------------------------
			Saved files:
			"""
			report += "\n" + "\n".join(f"                {filename}" for filename in saved_files)
		with open(f"log_{int(time.time())}.log", "x") as file:
			file.write(report)

if __name__ == '__main__':# 7
	# Previous benchmark entrypoint, kept intact for comparison:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # 2, 24 (and 2, 56) best, dict_size = 3 best
    oist_demo(dims = 5, hidden_layers = 1, hidden_dim = 64, dictionary_size = 4, repeats = 1, save_plots = True, save_report = True, device = 'cuda:0', epochs = 200, patience = 25, placeholder = True, poly_deg = 2, save_predictions = False, use_residual_penalty = False)
    # best so far, hidden_dim = 64, dict_size = 4
    # absolute best, 1 hidden layer, 64 hidden dims
    # tried 56 (good), 
    oist_demo(dims = 10, hidden_layers = 1, hidden_dim = 16, dictionary_size = 3, repeats = 1, save_plots = True, save_report = True, device = 'cuda:0', epochs = 200, patience = 25, placeholder = True, poly_deg = 2, save_predictions = False, use_residual_penalty = False)
	# reference_filename = 'reconstructed_baseline.csv'
	# prediction_filename = 'reconstructed_signal.csv'
	# error = reconstruction_relative_l2_error(reference_filename, prediction_filename, sample_fraction = 0.2, seed = 0)
	# print(f"Sampled relative L2 reconstruction error (20% of rows): {error:.6f}%")
    
    #oist_delay_hybrid_demo(
    #    dims = 30,
    #    dictionary_size = 60,
    #    hidden_layers = 2,
    #    hidden_dim = 12,
    #    n_lags = 6,
    #    repeats = 1,
    #    save_plots = True,
    #    save_report = True,
    #    device = device,
    #    epochs = 200,
    #    patience = 50,
    #    compare_polynomial = True,
    #    poly_deg = 1,
    #    save_predictions = False,
    #    seed = 0,
    #    test_size = 0.5,
    #    use_residual_penalty = True,
    #    shuffle = False,
    #    include_identity = True,
    #    include_constant = True
    #)
