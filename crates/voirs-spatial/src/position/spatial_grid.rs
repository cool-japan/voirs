//! Spatial grid for efficient proximity queries

use crate::types::Position3D;

/// Spatial grid for efficient proximity queries
pub struct SpatialGrid {
    /// Grid cell size in meters
    cell_size: f32,
    /// Grid dimensions
    grid_size: (usize, usize, usize),
    /// Grid cells containing source IDs
    cells: Vec<Vec<Vec<Vec<String>>>>,
    /// Grid bounds
    bounds: (Position3D, Position3D),
}

impl SpatialGrid {
    /// Create new spatial grid
    pub fn new(bounds: (Position3D, Position3D), cell_size: f32) -> Self {
        let (min_bounds, max_bounds) = bounds;
        let width = ((max_bounds.x - min_bounds.x) / cell_size).ceil() as usize;
        let height = ((max_bounds.y - min_bounds.y) / cell_size).ceil() as usize;
        let depth = ((max_bounds.z - min_bounds.z) / cell_size).ceil() as usize;

        let mut cells = Vec::with_capacity(width);
        for _ in 0..width {
            let mut column = Vec::with_capacity(height);
            for _ in 0..height {
                let mut layer = Vec::with_capacity(depth);
                for _ in 0..depth {
                    layer.push(Vec::new());
                }
                column.push(layer);
            }
            cells.push(column);
        }

        Self {
            cell_size,
            grid_size: (width, height, depth),
            cells,
            bounds,
        }
    }

    /// Add source to grid
    pub fn add_source(&mut self, source_id: &str, position: Position3D) {
        if let Some((x, y, z)) = self.position_to_cell(position) {
            self.cells[x][y][z].push(source_id.to_string());
        }
    }

    /// Remove source from grid
    pub fn remove_source(&mut self, source_id: &str) {
        for x in 0..self.grid_size.0 {
            for y in 0..self.grid_size.1 {
                for z in 0..self.grid_size.2 {
                    self.cells[x][y][z].retain(|id| id != source_id);
                }
            }
        }
    }

    /// Move source in grid
    pub fn move_source(
        &mut self,
        source_id: &str,
        old_position: Position3D,
        new_position: Position3D,
    ) {
        let old_cell = self.position_to_cell(old_position);
        let new_cell = self.position_to_cell(new_position);

        if old_cell != new_cell {
            // Remove from old cell
            if let Some((x, y, z)) = old_cell {
                self.cells[x][y][z].retain(|id| id != source_id);
            }

            // Add to new cell
            if let Some((x, y, z)) = new_cell {
                self.cells[x][y][z].push(source_id.to_string());
            }
        }
    }

    /// Query sources within sphere
    pub fn query_sphere(&self, center: Position3D, radius: f32) -> Vec<String> {
        let mut results = Vec::new();
        let radius_cells = (radius / self.cell_size).ceil() as i32;

        if let Some((cx, cy, cz)) = self.position_to_cell(center) {
            let cx = cx as i32;
            let cy = cy as i32;
            let cz = cz as i32;

            for dx in -radius_cells..=radius_cells {
                for dy in -radius_cells..=radius_cells {
                    for dz in -radius_cells..=radius_cells {
                        let x = cx + dx;
                        let y = cy + dy;
                        let z = cz + dz;

                        if x >= 0
                            && y >= 0
                            && z >= 0
                            && (x as usize) < self.grid_size.0
                            && (y as usize) < self.grid_size.1
                            && (z as usize) < self.grid_size.2
                        {
                            results.extend(
                                self.cells[x as usize][y as usize][z as usize]
                                    .iter()
                                    .cloned(),
                            );
                        }
                    }
                }
            }
        }

        results
    }

    /// Convert position to grid cell coordinates
    fn position_to_cell(&self, position: Position3D) -> Option<(usize, usize, usize)> {
        let (min_bounds, _) = self.bounds;

        let x = ((position.x - min_bounds.x) / self.cell_size) as usize;
        let y = ((position.y - min_bounds.y) / self.cell_size) as usize;
        let z = ((position.z - min_bounds.z) / self.cell_size) as usize;

        if x < self.grid_size.0 && y < self.grid_size.1 && z < self.grid_size.2 {
            Some((x, y, z))
        } else {
            None
        }
    }
}
