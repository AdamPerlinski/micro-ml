mod error;
mod linear;
mod polynomial;
mod exponential;

use wasm_bindgen::prelude::*;
pub use error::MlError;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

pub use linear::*;
pub use polynomial::*;
pub use exponential::*;
