/** new 模型 */
// 模型的数据
const modelData = [{
  label: '无',
  value: 'None',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/nones.png',
  decs: '未选择模型'
}, {
  label: '3D电影风格',
  value: '[SD1.5]3D电影风格.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/3dAnimationDiffusion_v10.png',
  decs: '迪斯尼、梦工厂、皮克斯'
}, {
  label: '洪恩绘本模型1',
  value: '[SDXL]洪恩绘本模型1.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/artbook_sdxl_v1.1.fp16.png',
  decs: 'painting,artbook'
}, {
  label: '洪恩绘本模型v2',
  value: '[SD1.5]洪恩绘本模型v2.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/artbookv2.png',
  decs: '儿童绘本'
}, {
  label: '二次元动漫',
  value: '[SDXL]二次元动漫.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/bluePencilXL_v310.png',
  decs: '二次元动漫风格'
}, {
  label: '皮克斯3D',
  value: '[SD1.5]皮克斯3D.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/disneyPixarCartoon_v10b.png',
  decs: '皮克斯3D建模风格'
}, {
  label: '迪士尼3D',
  value: '[SD1.5]迪士尼3D.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/disneyStyleV1_v10.png',
  decs: '[SD1.5] 迪士尼3D建模风格'
}, {
  label: 'DreamShaper8',
  value: '[SD1.5]DreamShaper8.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/DreamShaper_8_pruned.png',
  decs: '各类艺术风格形式'
}, {
  label: 'DreamShaper+盲盒',
  value: '[SD1.5]DreamShaper+盲盒.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/dreamshaper_blindbox_c4d.fp16.png',
  decs: 'Dreamshaper+盲盒c4dlora'
}, {
  label: '3D渲染',
  value: '[SDXL]3D渲染.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/fastercluck_v2.png',
  decs: '3D渲染风格'
}, {
  label: 'IP设计3D模型',
  value: '[SD1.5]IP设计3D模型.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/ipDESIGN3D_v31.png',
  decs: 'IP设计3D模型渲染风格'
}, {
  label: '盲盒公仔',
  value: '[SD1.5]盲盒公仔.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/helloip3dIP_helloip3dV13e.png',
  decs: '塑料盲盒公仔风格'
}, {
  label: 'Fooocus默认SDXL[旧]',
  value: '[SDXL]基础模型juggernautXLv8.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/juggernautXL_version6Rundiffusion.png',
  decs: '[SDXL]各类艺术风格,SDXL基础模型'
}, { 
  label: 'Fooocus默认SDXL',
  value: '[SDXL]基础模型juggernautXLvXNSFW.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/juggernautXL_version6Rundiffusion.png',
  decs: '[SDXL]各类艺术风格,SDXL基础模型'
}, {
  label: 'SDXL基础模型游乐场2',
  value: '[SDXL]基础模型游乐场2.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/playground-v2.fp16.png',
  decs: '[SDXL]各类艺术风格,SDXL基础模型'
}, {
  label: '真实电影v20',
  value: '[SDXL]真实电影v20.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/realisticStockPhoto_v10.png',
  decs: '真实,照片质感,电影,SDXL基础模型'
}, {
  label: '真实电影v10',
  value: '[SDXL]真实电影v10.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/realisticStockPhoto_v10.png',
  decs: '真实,照片质感,电影,SDXL基础模型'
}, {
  label: '基础二次元',
  value: '[SD1.5]revAnimated二次元.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/revAnimated_v122EOL.png',
  decs: '绘本,动漫,幻想,卡通'
}, {
  label: 'SDXL基础精炼模型VAE',
  value: '[SDXL]SDXL基础精炼模型VAE.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/sd_xl_base_1.0_0.9vae.png',
  decs: '真实/绘画细节填充,SDXL基础精炼模型'
}, {
  label: '原始SDXL模型',
  value: '[SDXL]原始SDXL模型1.0.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/sd_xl_base_1.0.png',
  decs: ' 原始SDXL基础模型'
}, {
  label: '可爱3D',
  value: '[SD1.5]sdvn53可爱3Dv10.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/sdvn53dcutewave_v10.png',
  decs: '3D,儿童,玩具,角色设计'
}, {
  label: '原始SDXL专属精炼',
  value: '[SDXL]原始SDXL基础专属精炼模型.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/sd_xl_refiner_1.0_0.9vae.png',
  decs: '原始SDXL基础精炼模型vae'
}, {
  label: '洪恩绘本SDXLv2',
  value: '[SDXL]洪恩绘本模型v2.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/sdxl_artbookv2_sdvn.png',
  decs: '手绘痕迹更明显'
}, {
  label: 'DreamShaper加速',
  value: '[SDXL]DreamShaperXL加速.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/DreamShaperXL_Turbo_dpmppSdeKarras_half_pruned_6.png',
  decs: 'DreamShaper SDXL turbo版本'
}, {
  label: 'AnimaPencil动漫',
  value: '[SDXL]AnimaPencil动漫v400.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/animaPencilXL_v100.png',
  decs: ' Fooocus预置包动漫二次元'
}, {
  label: '动漫与真实融合',
  value: '[SDXL]动漫+真实v20.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/albedobaseXL_v20.png',
  decs: ' Fooocus预置包融合'
}, {
  label: '网红拍照',
  value: '[SD1.5]网红麦橘写实.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/wanghongsd15.jpg',
  decs: '女生网红拍照'
}, {
  label: '平面动漫',
  value: '[SD1.5]平面动漫线条.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/flat2DAnimerge_v45Sharp.png',
  decs: '轮廓线平面色彩'
}, {
  label: 'BluePencil动漫',
  value: '[SDXL]BluePencil动漫.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/bluepencil.jpg',
  decs: '柔和色彩二次元动漫'
}, {
  label: '亚洲人像',
  value: '[SDXL]亚洲人像.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/asiaportrait.png',
  decs: '国风审美的人像'
}, {
  label: 'SD3Medium',
  value: 'sd3_medium_incl_clips.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/sd3.png',
  decs: 'SD3Mdedium'
}, {
  label: 'PlaygroundV2.5',
  value: 'Playground-v2.5.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/playgroundv2.5.png',
  decs: 'PlaygroundV2.5'
}, {
  label: 'NijiV6',
  value: '[SDXL]NijiV6.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/nijiv6_.jpeg',
  decs: 'NijiEX style'
}, {
  label: 'NijiV6+古诗',
  value: 'human_MidjourneyV1.2_base_sdxl.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/niji_ihunman.jpeg',
  decs: 'NijiEX style'
}]

// lora的数据
const loraData = [{
  label: '无',
  value: 'None',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/nones.png',
  decs: '未加载lora'
}, {
  label: '可爱皮克斯角色',
  value: '可爱3d.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/3dmodel_cute3d.png',
  decs: 'MG_ip'
}, {
  label: '3D渲染风格',
  value: '3D渲染.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/3DRedmond-3DRenderStyle-3DRenderAF.png',
  decs: '3D Render Style,3DRenderAF'
}, {
  label: '细化',
  value: '细节增修.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/add-detail-xl.png',
  decs: '细节化，添加，修改细节'
}, {
  label: '扁平艺术插图',
  value: '扁平艺术插图.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/chatuxuan.png',
  decs: 'chatu'
}, {
  label: '绘本儿童角色',
  value: '绘本儿童角色.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/COOLKIDS_XL_0.3_RC.png',
  decs: '手绘儿童绘本风格'
}, {
  label: '可爱盲盒',
  value: '可爱3D盲盒.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/cute_blindbox_sdxl.png',
  decs: 'blindbox'
}, {
  label: '动森场景风格',
  value: '动森场景.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/dongsen_xl.png',
  decs: 'chibi,3D'
}, {
  label: '手绘魔幻插图',
  value: '手绘魔幻插图.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/free-hand.png',
  decs: 'free-hand'
}, {
  label: '蜡笔油画',
  value: '蜡笔油画.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/crayon_paint.jpeg',
  decs: 'childpaint'
}, {
  label: '可爱动物',
  value: '可爱动物.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/loha_animal_sdxl.png',
  decs: '可爱，毛茸茸的小动物'
}, {
  label: '现代儿童绘本',
  value: '儿童绘本彩色插画.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/MODILL_XL_0.27_RC.png',
  decs: '现代儿童绘本彩色插画'
}, {
  label: '油墨广告报纸',
  value: '油墨广告.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/OldillXL_0.4_RC.png',
  decs: '老派欧式广告报纸插画'
}, {
  label: '3D卡通电影',
  value: 'Samaritan3D卡通电影.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/Samaritan 3d Cartoon SDXL.png',
  decs: '3D动漫,midjourney'
}, {
  label: 'SDXL基础lora',
  value: 'sdxl样例lora.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/sd_xl_offset_example-lora_1.0.jpeg',
  decs: 'SDXL基础lora,细节增修'
}, {
  label: '电影画质',
  value: '电影画质.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.jpeg',
  decs: '电影画质，真实，超清'
}, {
  label: 'lcm加速',
  value: 'sdxl_lcm加速.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/sdxl_lcm_lora.png',
  decs: 'lcm加速lora'
}, {
  label: '物品场景3D',
  value: '物品场景3D.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/Stylized_Setting_SDXL.png',
  decs: '游戏物品饰品/三维渲染建筑物，地图插件'
}, {
  label: '三视图lora',
  value: 'mw_3d角色ip三视图q版_2.0.1(XL尝鲜版).safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/mw_3d角色ip三视图q版_2.0.1(XL尝鲜版).png',
  decs: 'mw_sanshitu,three view,full body'
}, {
  label: '儿童绘本',
  value: '儿童绘本.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/Children_Illustration_SDXL.png',
  decs: 'Children\'s Illustration Style'
}, {
  label: 'AHAworld',
  value: 'AHAworld.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/ahaworld.png',
  decs: 'AHA世界角色风格'
}, {
  label: 'AppIcons',
  value: 'appicons.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/appicons.png',
  decs: 'APP类型图标'
}, {
  label: '儿童插画风格',
  value: '儿童插画风格.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/J_drawing_XL.jpeg',
  decs: 'j_drawing'
}, {
  label: '纪念碑谷风格',
  value: '纪念碑谷风格.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/J_drawing_XL.jpeg',
  decs: 'video game Monument Valley style'
}, {
  label: '川剧变脸风格',
  value: '川剧变脸.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/chuanju_bianlian.png',
  decs: 'sichuan opera'
}, {
  label: '粘土风格',
  value: '粘土风格.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/claymation.jpeg',
  decs: 'claymation'
}, {
  label: 'Q版盲盒',
  value: 'Q版3D盲盒SDXL_v1.0.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/qver_3d_blindbox.jpg',
  decs: 'owo style, chibi'
}, {
  label: 'XL盲盒',
  value: 'XL-Designer_BlindBox_v1.0.safetensors',
  image: 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/xl_designer_blindbox.webp',
  decs: 'chibi'
}]

// 点击回显的数据
let chooseData = {
  baseModel: modelData[0],
  lora: {}
}

// 弹框类型
const MODEL = 'model'
const LORA_ITEM = 'lora-item-'
const REFINER_MODEL = 'refiner-model'
let dialogType = MODEL // | lora-item-index

/** 获取元素 */
function getElement(id) {
  return gradioApp().getElementById(id);
}

/** 创建对应模型元素 基础模型*/
async function createModelElm() {
  // 最上层的原始div
  const baseModel = getElement('base_model')
  const oldModel = baseModel.querySelectorAll('.wrap-inner');
  // if (oldModel && oldModel.length !== 0) {
  //   oldModel[0].parentElement.style.display = 'none'
  // }
  // 判断是否已经存在这个元素
  let componentShowModel = null
  if (!getElement('component-show-model')) {
    // 更换的新的div-父级
    componentShowModel = document.createElement('div');
    componentShowModel.id = "component-show-model";
    baseModel.appendChild(componentShowModel);
  } else {
    resetElItem(MODEL)
    // resetBtnItem(REFINER_MODEL)
    componentShowModel = getElement('component-show-model');
  }
  const inputs = oldModel[0].querySelectorAll('input');
  const findItem = modelData.find((i) => i.value === inputs[0].value)
  const modelChoose = showModelChoose(findItem, MODEL)
  componentShowModel.append(modelChoose);
}
/** 创建对应模型元素 精炼模型*/
async function createModelRefinerElm() {
  // 最上层的原始div
  const baseModel = getElement('refiner_model')
  const oldModel = baseModel.querySelectorAll('.wrap-inner');
  // if (oldModel && oldModel.length !== 0) {
  //   oldModel[0].parentElement.style.display = 'none'
  // }

  let componentShowModel = null
  if (!getElement('component-show-refiner-model')) {
    // 更换的新的div-父级
    componentShowModel = document.createElement('div');
    componentShowModel.id = "component-show-refiner-model";
    baseModel.appendChild(componentShowModel);
  } else {
    resetElItem(REFINER_MODEL)
    // resetBtnItem(REFINER_MODEL)
    componentShowModel = getElement('component-show-refiner-model');
  }
  const inputs = oldModel[0].querySelectorAll('input');
  if (inputs[0] && inputs[0].value !== 'None') {
    const findItem = modelData.find((i) => i.value === inputs[0].value)
    const modelChoose = showModelChoose(findItem, REFINER_MODEL)
    componentShowModel.append(modelChoose);
  }
  //  else {
  //   btnNone = noChooseBtnItem(REFINER_MODEL)
  //   componentShowModel.append(btnNone);
  // }
}
// 没有选中时的 点击按钮
function noChooseBtnItem(type) {
  const btnNone = document.createElement('button');
  btnNone.id = "choose-div-button-" + type;
  btnNone.style.width = '100px';
  btnNone.style.padding = "10px";
  btnNone.style.borderRadius = "8px";
  btnNone.style.fontSize = "12px";
  // btnNone.textContent = '点击选择';
  // btnNone.style.border = "1px solid #374151";


  // 为最外层元素添加点击事件监听器
  // btnNone.addEventListener("click", function () {
  //   dialogType = type
  //   chooseData.baseModel = modelData[0]
  //   toChooseModel()
  // });
  return btnNone
}

// 没有选中时默认的lora按钮
function showLoRaAllWrapNoChoose() {
  const baseloRaWrap = getElement('LoRA-All-Group')
  const oldModel = baseloRaWrap.querySelectorAll('.wrap-inner');
  for (let index = 0; index < oldModel.length; index++) {
    let loRasAllWrap = null
    if (!getElement("component-wrap-lora-item-" + index)) {
      // 更换的新的div-父级
      loRasAllWrap = document.createElement('div');
      loRasAllWrap.id = "component-wrap-lora-item-" + index;
      oldModel[index].parentElement.parentElement.appendChild(loRasAllWrap);
    } else {
      resetElItem(LORA_ITEM + index)
      // resetBtnItem(LORA_ITEM + index)
      loRasAllWrap = getElement("component-wrap-lora-item-" + index);
    }
    // 查找对应input的值
    const oldModelInput = oldModel[index].querySelectorAll('input');
    // 有值就展示回显 没有就 按钮
    if (oldModelInput[0] && oldModelInput[0].value !== 'None') {
      const findItem = loraData.find((i) => i.value === oldModelInput[0].value)
      const modelChoose = showModelChoose(findItem, LORA_ITEM + index)
      loRasAllWrap.appendChild(modelChoose);
      oldModel[index].parentElement.parentElement.appendChild(loRasAllWrap);
    }
    // else {
    //   btnNone = noChooseBtnItem(LORA_ITEM + index)
    //   loRasAllWrap.append(btnNone);
    // }
    // oldModel[index].parentElement.style.display = 'none'
  }
}

// 选中回显的模型 
function showModelChoose(params, type) {
  // 回显设置的模型
  const modelChoose = document.createElement('button');
  modelChoose.id = "choose-div-" + type;
  modelChoose.style.width = '100%';
  const chooseItemDiv = showOnChooseModelItem(params)
  modelChoose.append(chooseItemDiv);

  // 为最外层元素添加点击事件监听器
  // modelChoose.addEventListener("click", function () {
  //   dialogType = type
  //   chooseData.baseModel = params
  //   toChooseModel()
  // });
  return modelChoose
}
function showOnChooseModelItem(chooseData) {
  // 点击事件的回显的模型 modelChoose的子级
  const chooseItemDiv = document.createElement('div');
  chooseItemDiv.style.padding = "10px";
  chooseItemDiv.style.border = "1px solid #374151";
  chooseItemDiv.style.borderRadius = "8px";
  chooseItemDiv.style.fontSize = "12px";
  chooseItemDiv.style.display = "flex";
  chooseItemDiv.style.justifyContent = "flex-start";
  chooseItemDiv.style.alignItems = "center";
  chooseItemDiv.style.width = '100%';

  // 选中模型的略缩图 chooseItemDiv的子级
  const itemImage = document.createElement('img');
  itemImage.src = chooseData.image;
  itemImage.style.width = '50px';
  itemImage.style.height = '50px';

  const textWrapper = document.createElement('div');
  textWrapper.style.flex = '1'; // 占据剩余空间
  textWrapper.style.minWidth = '0'; // 防止 flex 子项溢出父容器
  textWrapper.id = "choose-model-item-text";
  textWrapper.style.marginLeft = "10px";
  textWrapper.style.textAlign = "left";
  // 选中模型的描述 chooseItemDiv的子级
  const itemLabel = document.createElement('div');
  itemLabel.innerHTML = chooseData.label;
  itemLabel.style.flex = '1'; // 让文字容器占据剩余空间
  itemLabel.style.whiteSpace = 'nowrap'; // 防止文字换行
  itemLabel.style.overflow = 'hidden'; // 隐藏超出部分
  itemLabel.style.textOverflow = 'ellipsis'; // 超出部分显示省略号
  // 选中模型的描述 chooseItemDiv的子级
  const itemDesc = document.createElement('div');
  itemDesc.innerHTML = chooseData.decs;
  itemDesc.style.marginTop = "8px";
  itemDesc.style.color = "#9ca3af";
  itemDesc.style.flex = '1'; // 让文字容器占据剩余空间
  itemDesc.style.whiteSpace = 'nowrap'; // 防止文字换行
  itemDesc.style.overflow = 'hidden'; // 隐藏超出部分
  itemDesc.style.textOverflow = 'ellipsis'; // 超出部分显示省略号

  // 插入所有的元素
  textWrapper.append(itemLabel);
  textWrapper.append(itemDesc);
  chooseItemDiv.append(itemImage);
  chooseItemDiv.append(textWrapper);

  return chooseItemDiv
}

const mask = document.createElement('div');
// 点击出现的弹框
function toChooseModel() {
  console.log("选中的模版被点击了！");
  showDialog()
}
// 创建遮罩层
function showDialogTask() {
  document.body.appendChild(mask);
  mask.style.position = 'fixed';
  mask.style.top = '0';
  mask.style.left = '0';
  mask.style.width = '100%';
  mask.style.height = '100%';
  mask.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
  mask.style.display = 'flex';
  mask.style.justifyContent = 'center';
  mask.style.alignItems = 'center';
  mask.style.zIndex = '1000'; // 确保遮罩层在顶层
  mask.style.visibility = 'hidden'; // 初始状态不可见
}

// 创建弹框
function showDialogContent(params) {
  const dialogData = (dialogType === MODEL || dialogType === REFINER_MODEL) ? modelData : loraData
  const dialog = document.createElement('div');
  dialog.id = 'dialog-wrap';
  dialog.style.backgroundColor = '#374151';
  dialog.style.width = '60%';
  dialog.style.minHeight = '200px';
  dialog.style.padding = '10px';
  dialog.style.borderRadius = '5px';
  dialog.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.3)';
  mask.appendChild(dialog);

  // 弹框标题
  const dialogTitleContent = document.createElement('div');
  dialogTitleContent.id = 'dialog-title-content'
  dialogTitleContent.style = 'padding: 10px; display: flex; justify-content: space-between; align-items: center; width: 98%; color: #fff'
  dialogTitleContent.style.borderBottom = '1px solid rgba(255,255,255,0.5)'
  dialogTitleContent.innerHTML = '<div style="">' + params.dialogTitleName + '</div>';
  // 关闭按钮
  const closeButton = document.createElement('div');
  closeButton.textContent = 'X';
  dialogTitleContent.appendChild(closeButton);
  dialog.appendChild(dialogTitleContent);
  // 为关闭按钮添加事件监听器
  // closeButton.addEventListener('click', hideDialog);

  // 弹框内容
  const dialogContent = document.createElement('div');
  dialogContent.id = 'dialog-body-content';
  dialogContent.style.display = 'flex';
  dialogContent.style.flexWrap = 'wrap';
  dialogContent.style.gap = '8px'; // 设置盒子之间的间隔
  dialogContent.style.paddingTop = "15px";
  dialogContent.style.height = "600px";
  dialogContent.style.overflow = "auto";

  for (let index = 0; index < dialogData.length; index++) {
    const itemModelContent = modelItemContent(dialogData[index])
    dialogContent.appendChild(itemModelContent)
    // itemModelContent.addEventListener('click', function () {
    //   chooseData.baseModel = JSON.parse(this.value)
    //   if (dialogType === MODEL) {
    //     setChooseModelValue('choose-div-model', chooseData.baseModel, dialogType)
    //   } else if (dialogType.includes(LORA_ITEM)) {
    //     setChooseLoraItemValue(chooseData.baseModel, dialogType)
    //   } else if (dialogType === REFINER_MODEL) {
    //     setChooseRefinerModelValue('choose-div-refiner-model', chooseData.baseModel, dialogType)
    //   }

    // })
  }

  dialog.appendChild(dialogContent);
}

// 显示遮罩层和弹框
function showDialog() {
  mask.style.visibility = 'visible';
  showDialogContent({ dialogTitleName: '目录' })
}

// 隐藏遮罩层和弹框
function hideDialog() {
  mask.style.visibility = 'hidden';
  const dialogEl = getElement('dialog-wrap')
  mask.removeChild(dialogEl)
}

// 弹框中展示的模型Item
function modelItemContent(params) {
  // 最外层的div
  const itemModelContent = document.createElement('div');
  itemModelContent.id = 'item-model-content';
  itemModelContent.style.backgroundColor = '#374151';
  itemModelContent.style.width = '140px';
  itemModelContent.style.minHeight = '200px';
  itemModelContent.style.padding = '10px';
  itemModelContent.style.borderRadius = '5px';
  itemModelContent.style.border = params.value === chooseData.baseModel.value ? '2px solid #409EFF' : '1px solid rgba(255,255,255,0.5)'
  itemModelContent.style.fontSize = '12px'
  itemModelContent.style.color = '#fff'
  itemModelContent.style.marginTop = '10px';
  itemModelContent.value = JSON.stringify(params)
  // 选中模型的略缩图 chooseItemDiv的子级
  const itemImage = document.createElement('img');
  itemImage.src = params.image;
  itemImage.style.width = '140px';
  itemImage.style.height = '140px';
  const itemTextContent = document.createElement('div');
  itemTextContent.id = "model-item-content";

  // 选中模型的描述 chooseItemDiv的子级
  const itemLabel = document.createElement('div');
  itemLabel.innerHTML = params.label;
  itemLabel.style.marginTop = "10px";
  itemLabel.style.width = '140px';
  itemLabel.style.overflow = "hidden";
  itemLabel.style.textOverflow = "ellipsis";
  itemLabel.style.whiteSpace = "nowrap";

  // 选中模型的描述 chooseItemDiv的子级
  const itemDesc = document.createElement('div');
  itemDesc.innerHTML = params.decs;
  itemDesc.style.width = '140px';
  itemDesc.style.textOverflow = "ellipsis";
  itemDesc.style.whiteSpace = "wrap";

  itemTextContent.appendChild(itemLabel)
  itemTextContent.appendChild(itemDesc)
  itemModelContent.appendChild(itemImage)
  itemModelContent.appendChild(itemTextContent)

  return itemModelContent
}




// 查找 对应的 input 和 需要回显的div  进行回显 === 基础模型
function setChooseModelValue(baseElId, params, type) {
  const componentShowModel = getElement('component-show-model')
  resetBtnItem(type)
  resetElItem(type)
  if (params.value === 'None') {
    btnNone = noChooseBtnItem(type)
    componentShowModel.append(btnNone);
  } else {
    const modelChoose = showModelChoose(params, type)
    componentShowModel.append(modelChoose);
  }
  // 给input 赋值 
  setInputValue('base_model', params, 0)
  // 隐藏dialog
  hideDialog()
}
// 查找 对应的 input 和 需要回显的div  进行回显 === 精炼模型
function setChooseRefinerModelValue(baseElId, params, type) {
  const componentShowModel = getElement('component-show-refiner-model')
  const refinerSwitch = getElement('refiner_switch')
  resetBtnItem(type)
  resetElItem(type)
  if (params.value === 'None') {
    btnNone = noChooseBtnItem(type)
    componentShowModel.append(btnNone);
    // 隐藏 精炼开启位置
    if (refinerSwitch) {
      refinerSwitch.style.display = 'none'
      refinerSwitch.parentNode.style.display = 'none'
    }
  } else {
    const modelChoose = showModelChoose(params, type)
    componentShowModel.append(modelChoose);
    // 显示 精炼开启位置
    if (refinerSwitch) {
      refinerSwitch.style.display = 'block'
      refinerSwitch.parentNode.style.display = 'block'
    }
  }
  // 给input 赋值 
  setInputValue('refiner_model', params, 0)
  // 隐藏dialog
  hideDialog()
}
// 查找 对应的 input 和 需要回显的div  进行回显 === lora
function setChooseLoraItemValue(params, type) {
  // 创建一个新的lora
  const index = Math.round(type.split('-')[2])
  const baseloRaWrap = getElement('LoRA-All-Group')
  const oldModel = baseloRaWrap.querySelectorAll('.wrap-inner');
  const loRasAllWrap = getElement('component-wrap-' + type);
  resetBtnItem(type)
  resetElItem(type)
  if (params.value === 'None') {
    btnNone = noChooseBtnItem(type)
    loRasAllWrap.appendChild(btnNone);
  } else {
    const modelChoose = showModelChoose(params, LORA_ITEM + index)
    loRasAllWrap.appendChild(modelChoose);
  }
  oldModel[index].parentElement.parentElement.appendChild(loRasAllWrap);
  setInputValue('LoRA-All-Group', params, index)
  // 隐藏dialog
  hideDialog()
}

function resetElItem(type) {
  // 回显选中的模型
  const element = getElement('choose-div-' + type);
  if (element) {
    element.parentNode.removeChild(element);
  }
}

function resetBtnItem(type) {
  // 选中了  删除点击选择按钮
  const btn = getElement('choose-div-button-' + type)
  if (btn) {
    btn.parentNode.removeChild(btn);
  }
}
// 给input 赋值 
function setInputValue(baseElId, params, index) {
  const baseModel = getElement(baseElId)
  const oldModel = baseModel.querySelectorAll('.wrap-inner');
  if (oldModel && oldModel.length !== 0) {
    // 查找该元素下的所有input元素
    const inputs = oldModel[index].querySelectorAll('input');
    // 遍历所有找到的input元素并赋值
    inputs.forEach(function (input) {
      input.value = params.value;
      updateInput(input)
    });
  }
}

function initNewModel() {
  createModelElm()
  createModelRefinerElm()
  showLoRaAllWrapNoChoose()
}

onUiLoaded(async () => {
  // showDialogTask()
  initNewModel()
})

