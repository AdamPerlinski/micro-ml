mod error;
mod linear;
mod polynomial;
mod exponential;
mod timeseries;

use wasm_bindgen::prelude::*;
pub use error::MlError;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// Re-export all public types and functions
pub use linear::*;
pub use polynomial::*;
pub use exponential::*;
pub use timeseries::*;
