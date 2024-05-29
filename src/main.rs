use std::sync::Arc;

use wgpu::{
    Color, CommandEncoderDescriptor, Device, DeviceDescriptor, Instance, InstanceDescriptor,
    Operations, Queue, RenderPassColorAttachment, RenderPassDescriptor, RequestAdapterOptions,
    Surface, SurfaceConfiguration, SurfaceError, TextureUsages, TextureViewDescriptor,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

fn initialise_webgpu<'a>(
    window: Arc<Window>,
) -> Result<(Surface<'a>, SurfaceConfiguration, Device, Queue), StateError> {
    let instance = Instance::new(InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone())?;
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        compatible_surface: Some(&surface),
        ..Default::default()
    }))
    .ok_or(StateError::NoAdapterFound)?;

    let (device, queue) =
        pollster::block_on(adapter.request_device(&DeviceDescriptor::default(), None))?;

    let surface_capabilities = surface.get_capabilities(&adapter);
    let surface_format = surface_capabilities
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_capabilities.formats[0]);

    let size = window.inner_size();

    let surface_config = SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: surface_capabilities.present_modes[0],
        desired_maximum_frame_latency: 2,
        alpha_mode: surface_capabilities.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(&device, &surface_config);

    Ok((surface, surface_config, device, queue))
}

#[derive(thiserror::Error, Debug)]
enum StateError {
    #[error("Failed to create surface: {0}")]
    CreateSurfaceError(#[from] wgpu::CreateSurfaceError),
    #[error("Requesting a device has failed: {0}")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error("No adapter was found")]
    NoAdapterFound,
}

struct State<'a> {
    surface: Surface<'a>,
    surface_config: SurfaceConfiguration,
    window: Arc<Window>,
    device: Device,
    queue: Queue,
    size: PhysicalSize<u32>,
    clear_color: Color,
}

impl<'a> State<'a> {
    fn new(window: Arc<Window>) -> Result<Self, StateError> {
        let size = window.inner_size();

        let (surface, surface_config, device, queue) = initialise_webgpu(window.clone())?;

        Ok(Self {
            surface,
            surface_config,
            window,
            device,
            queue,
            size,
            clear_color: Color {
                r: 0.0,
                g: 0.05,
                b: 0.2,
                a: 1.0,
            },
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.size = new_size;
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        {
            #[allow(unused_mut, unused_variables)]
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }

        self.queue.submit([encoder.finish()]);
        output.present();

        Ok(())
    }
}

struct App<'a> {
    title: &'static str,
    state: Option<State<'a>>,
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title(self.title))
                .unwrap(),
        );
        window.request_redraw();

        self.state = Some(State::new(window).unwrap());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        if window_id != state.window.id() {
            return;
        }

        if state.input(&event) {
            return;
        }

        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => {
                self.state = None;
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(SurfaceError::Lost) => state.resize(state.size),
                    Err(SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{:?}", e),
                }
                state.window.request_redraw();
            }
            WindowEvent::Resized(new_size) => state.resize(new_size),
            _ => (),
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App {
        title: "App",
        state: None,
    };
    event_loop.run_app(&mut app).unwrap();
}
