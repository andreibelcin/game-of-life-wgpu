use std::{
    ops::BitXor,
    sync::Arc,
    time::{Duration, Instant},
};

use rand::prelude::*;
use vertex::{create_vertex_buffer, get_vertex_buffer_layout, VERTICES};
use wgpu::{
    core::binding_model::BindGroupLayout,
    include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendState, Buffer, BufferDescriptor,
    BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor,
    FragmentState, Instance, InstanceDescriptor, MultisampleState, Operations,
    PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, PrimitiveState, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    RequestAdapterOptions, ShaderStages, Surface, SurfaceConfiguration, SurfaceError,
    TextureUsages, TextureViewDescriptor, VertexState,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

mod vertex;

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
    // Window
    window: Arc<Window>,
    // WebGPU components
    surface: Surface<'a>,
    surface_config: SurfaceConfiguration,
    device: Device,
    queue: Queue,
    // Render pipeline
    vertex_buffer: Buffer,
    bind_groups: [BindGroup; 2],
    cell_pipeline: RenderPipeline,
    simulation_pipeline: ComputePipeline,
    // Other
    size: PhysicalSize<u32>,
    clear_color: Color,
    grid_size: usize,
    selected_bind: usize,
    last_time: Instant,
    compute_delay: Duration,
}

impl<'a> State<'a> {
    fn new(window: Arc<Window>) -> Result<Self, StateError> {
        let size = window.inner_size();
        let grid_size = 512;

        let (surface, surface_config, device, queue) = initialise_webgpu(window.clone())?;

        let vertex_buffer =
            create_vertex_buffer(&device, (VERTICES.len() * std::mem::size_of::<f32>()) as _);
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&VERTICES));

        let vertex_buffer_layout = get_vertex_buffer_layout();

        let uniform_array = [grid_size as f32; 2];
        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&uniform_array),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let mut cell_states = vec![0.0f32; grid_size * grid_size];
        let cell_states_buffers = [
            device.create_buffer(&BufferDescriptor {
                label: None,
                size: (cell_states.len() * std::mem::size_of::<f32>()) as _,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device.create_buffer(&BufferDescriptor {
                label: None,
                size: (cell_states.len() * std::mem::size_of::<f32>()) as _,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ];

        let mut rng = rand::thread_rng();
        for i in 0..cell_states.len() {
            cell_states[i] = if rng.gen::<f32>() > 0.6 { 1.0 } else { 0.0 };
        }
        queue.write_buffer(
            &cell_states_buffers[0],
            0,
            bytemuck::cast_slice(&cell_states),
        );

        let shader = device.create_shader_module(include_wgsl!("shader.wgsl"));
        let simulator = device.create_shader_module(include_wgsl!("compute.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_groups = [
            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(
                            uniform_buffer.as_entire_buffer_binding(),
                        ),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(
                            cell_states_buffers[0].as_entire_buffer_binding(),
                        ),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(
                            cell_states_buffers[1].as_entire_buffer_binding(),
                        ),
                    },
                ],
            }),
            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(
                            uniform_buffer.as_entire_buffer_binding(),
                        ),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(
                            cell_states_buffers[1].as_entire_buffer_binding(),
                        ),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(
                            cell_states_buffers[0].as_entire_buffer_binding(),
                        ),
                    },
                ],
            }),
        ];

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let cell_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            vertex: VertexState {
                module: &shader,
                entry_point: "vertexMain",
                buffers: &[vertex_buffer_layout],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fragmentMain",
                targets: &[Some(ColorTargetState {
                    format: surface_config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            layout: Some(&pipeline_layout),
            primitive: PrimitiveState::default(),
            label: None,
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });
        let simulation_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &simulator,
            entry_point: "main",
            compilation_options: Default::default(),
        });

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
            vertex_buffer,
            cell_pipeline,
            simulation_pipeline,
            grid_size,
            bind_groups,
            selected_bind: 0,
            last_time: Instant::now(),
            compute_delay: Duration::from_millis(8),
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

    fn update(&mut self) {
        let now = Instant::now();
        if now - self.last_time < self.compute_delay {
            return;
        }

        self.last_time = now;

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

            compute_pass.set_pipeline(&self.simulation_pipeline);
            compute_pass.set_bind_group(0, &self.bind_groups[self.selected_bind], &[]);

            let workgroup_count = (self.grid_size as f32 / 8.0).ceil() as _;
            compute_pass.dispatch_workgroups(workgroup_count, workgroup_count, 1);
        }

        self.queue.submit([encoder.finish()]);

        self.selected_bind ^= 1;
    }

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

            render_pass.set_pipeline(&self.cell_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_groups[self.selected_bind], &[]);
            render_pass.draw(
                0..(VERTICES.len() as u32 / 2),
                0..(self.grid_size * self.grid_size) as _,
            );
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
